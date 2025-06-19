# File: backend/api/endpoints/designs.py
"""
Design Management API Endpoints
==============================
FastAPI endpoints for visual design CRUD operations
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, delete, update, func
from typing import List, Optional, Dict, Any
from uuid import UUID
import uuid
from datetime import datetime, timedelta

# Import models and schemas
from backend.models.database_models import Design, DesignNode, DesignConnection, ExecutionSession
from backend.schemas.api_schemas import (
    DesignCreate, DesignUpdate, DesignResponse, DesignListResponse,
    NodeCreate, NodeUpdate, NodeResponse,
    ConnectionCreate, ConnectionUpdate, ConnectionResponse,
    ExecutionConfig, ExecutionResponse, BaseResponse, ErrorResponse
)
from backend.database.connection import get_db_session
from backend.auth.dependencies import get_current_user

# Create router
router = APIRouter(prefix="/api/designs", tags=["designs"])

# ============================================================================
# DESIGN CRUD OPERATIONS
# ============================================================================

@router.post("/", response_model=DesignResponse, status_code=201)
async def create_design(
    design_data: DesignCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user)
):
    """Create a new visual design"""
    try:
        # Create design
        design = Design(
            name=design_data.name,
            description=design_data.description,
            version=design_data.version,
            canvas_settings=design_data.canvas_settings,
            created_by=UUID(current_user["id"]) if "id" in current_user else None
        )
        
        db.add(design)
        await db.commit()
        await db.refresh(design)
        
        # Convert to response format
        response_data = design.to_dict()
        response_data["node_count"] = 0
        response_data["connection_count"] = 0
        
        return DesignResponse(**response_data)
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create design: {str(e)}")

@router.get("/", response_model=DesignListResponse)
async def list_designs(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    search: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    db: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user)
):
    """List all designs with pagination and filtering"""
    try:
        # Build query
        query = select(Design)
        
        # Apply filters
        if search:
            query = query.where(Design.name.ilike(f"%{search}%"))
        if is_active is not None:
            query = query.where(Design.is_active == is_active)
        
        # Add pagination
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        query = query.order_by(Design.updated_at.desc())
        
        # Execute query
        result = await db.execute(query)
        designs = result.scalars().all()
        
        # Get total count
        count_query = select(func.count(Design.id))
        if search:
            count_query = count_query.where(Design.name.ilike(f"%{search}%"))
        if is_active is not None:
            count_query = count_query.where(Design.is_active == is_active)
        
        total_result = await db.execute(count_query)
        total_count = total_result.scalar()
        
        # Convert to response format
        design_responses = []
        for design in designs:
            response_data = design.to_dict()
            
            # Get node and connection counts
            node_count_query = select(func.count(DesignNode.id)).where(DesignNode.design_id == design.id)
            node_count_result = await db.execute(node_count_query)
            response_data["node_count"] = node_count_result.scalar() or 0
            
            conn_count_query = select(func.count(DesignConnection.id)).where(DesignConnection.design_id == design.id)
            conn_count_result = await db.execute(conn_count_query)
            response_data["connection_count"] = conn_count_result.scalar() or 0
            
            design_responses.append(DesignResponse(**response_data))
        
        return DesignListResponse(
            designs=design_responses,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list designs: {str(e)}")

@router.get("/{design_id}", response_model=DesignResponse)
async def get_design(
    design_id: UUID = Path(..., description="Design ID"),
    include_details: bool = Query(False, description="Include nodes and connections"),
    db: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user)
):
    """Get a specific design by ID"""
    try:
        # Build query with optional relationships
        query = select(Design).where(Design.id == design_id)
        
        if include_details:
            query = query.options(
                selectinload(Design.nodes),
                selectinload(Design.connections)
            )
        
        result = await db.execute(query)
        design = result.scalar_one_or_none()
        
        if not design:
            raise HTTPException(status_code=404, detail="Design not found")
        
        # Convert to response format
        response_data = design.to_dict()
        
        if include_details:
            response_data["nodes"] = [node.to_dict() for node in design.nodes]
            response_data["connections"] = [conn.to_dict() for conn in design.connections]
            response_data["node_count"] = len(design.nodes)
            response_data["connection_count"] = len(design.connections)
        else:
            # Get counts
            node_count_query = select(func.count(DesignNode.id)).where(DesignNode.design_id == design.id)
            node_count_result = await db.execute(node_count_query)
            response_data["node_count"] = node_count_result.scalar() or 0
            
            conn_count_query = select(func.count(DesignConnection.id)).where(DesignConnection.design_id == design.id)
            conn_count_result = await db.execute(conn_count_query)
            response_data["connection_count"] = conn_count_result.scalar() or 0
        
        return DesignResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get design: {str(e)}")

@router.put("/{design_id}", response_model=DesignResponse)
async def update_design(
    design_id: UUID,
    design_data: DesignUpdate,
    db: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user)
):
    """Update a design"""
    try:
        # Get existing design
        query = select(Design).where(Design.id == design_id)
        result = await db.execute(query)
        design = result.scalar_one_or_none()
        
        if not design:
            raise HTTPException(status_code=404, detail="Design not found")
        
        # Update fields
        update_data = design_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(design, field, value)
        
        design.updated_at = datetime.now()
        
        await db.commit()
        await db.refresh(design)
        
        # Convert to response format
        response_data = design.to_dict()
        
        # Get counts
        node_count_query = select(func.count(DesignNode.id)).where(DesignNode.design_id == design.id)
        node_count_result = await db.execute(node_count_query)
        response_data["node_count"] = node_count_result.scalar() or 0
        
        conn_count_query = select(func.count(DesignConnection.id)).where(DesignConnection.design_id == design.id)
        conn_count_result = await db.execute(conn_count_query)
        response_data["connection_count"] = conn_count_result.scalar() or 0
        
        return DesignResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update design: {str(e)}")

@router.delete("/{design_id}", response_model=BaseResponse)
async def delete_design(
    design_id: UUID,
    force: bool = Query(False, description="Force delete even if executions exist"),
    db: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user)
):
    """Delete a design"""
    try:
        # Check if design exists
        query = select(Design).where(Design.id == design_id)
        result = await db.execute(query)
        design = result.scalar_one_or_none()
        
        if not design:
            raise HTTPException(status_code=404, detail="Design not found")
        
        # Check for active executions
        if not force:
            exec_query = select(ExecutionSession).where(
                ExecutionSession.design_id == design_id,
                ExecutionSession.status.in_(["running", "pending"])
            )
            exec_result = await db.execute(exec_query)
            active_executions = exec_result.scalars().all()
            
            if active_executions:
                raise HTTPException(
                    status_code=400, 
                    detail="Cannot delete design with active executions. Use force=true to override."
                )
        
        # Delete design (cascade will handle nodes and connections)
        await db.delete(design)
        await db.commit()
        
        return BaseResponse(message="Design deleted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete design: {str(e)}")

@router.post("/{design_id}/duplicate", response_model=DesignResponse)
async def duplicate_design(
    design_id: UUID,
    new_name: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user)
):
    """Duplicate an existing design"""
    try:
        # Get original design with all relationships
        query = select(Design).where(Design.id == design_id).options(
            selectinload(Design.nodes),
            selectinload(Design.connections)
        )
        result = await db.execute(query)
        original_design = result.scalar_one_or_none()
        
        if not original_design:
            raise HTTPException(status_code=404, detail="Design not found")
        
        # Create new design
        new_design = Design(
            name=new_name or f"{original_design.name} (Copy)",
            description=original_design.description,
            version="1.0.0",  # Reset version for copy
            canvas_settings=original_design.canvas_settings.copy(),
            created_by=UUID(current_user["id"]) if "id" in current_user else None
        )
        
        db.add(new_design)
        await db.flush()  # Get the new ID
        
        # Duplicate nodes
        node_id_mapping = {}
        for original_node in original_design.nodes:
            new_node_id = f"{original_node.node_id}_copy"
            new_node = DesignNode(
                design_id=new_design.id,
                node_id=new_node_id,
                component_type=original_node.component_type,
                position_x=original_node.position_x + 20,  # Slight offset
                position_y=original_node.position_y + 20,
                width=original_node.width,
                height=original_node.height,
                properties=original_node.properties.copy(),
                visual_style=original_node.visual_style.copy()
            )
            node_id_mapping[original_node.node_id] = new_node_id
            db.add(new_node)
        
        # Duplicate connections
        for original_conn in original_design.connections:
            new_from_id = node_id_mapping.get(original_conn.from_node_id)
            new_to_id = node_id_mapping.get(original_conn.to_node_id)
            
            if new_from_id and new_to_id:  # Only create if both nodes exist
                new_connection = DesignConnection(
                    design_id=new_design.id,
                    connection_id=f"{original_conn.connection_id}_copy",
                    from_node_id=new_from_id,
                    to_node_id=new_to_id,
                    from_port=original_conn.from_port,
                    to_port=original_conn.to_port,
                    connection_type=original_conn.connection_type,
                    properties=original_conn.properties.copy(),
                    visual_style=original_conn.visual_style.copy()
                )
                db.add(new_connection)
        
        await db.commit()
        await db.refresh(new_design)
        
        # Convert to response format
        response_data = new_design.to_dict()
        response_data["node_count"] = len(original_design.nodes)
        response_data["connection_count"] = len(original_design.connections)
        
        return DesignResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to duplicate design: {str(e)}")

# ============================================================================
# NODE OPERATIONS
# ============================================================================

@router.post("/{design_id}/nodes", response_model=NodeResponse, status_code=201)
async def create_node(
    design_id: UUID,
    node_data: NodeCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user)
):
    """Add a node to a design"""
    try:
        # Verify design exists
        design_query = select(Design).where(Design.id == design_id)
        design_result = await db.execute(design_query)
        design = design_result.scalar_one_or_none()
        
        if not design:
            raise HTTPException(status_code=404, detail="Design not found")
        
        # Check if node_id already exists in this design
        existing_query = select(DesignNode).where(
            DesignNode.design_id == design_id,
            DesignNode.node_id == node_data.node_id
        )
        existing_result = await db.execute(existing_query)
        if existing_result.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Node ID already exists in this design")
        
        # Create node
        node = DesignNode(
            design_id=design_id,
            node_id=node_data.node_id,
            component_type=node_data.component_type,
            position_x=node_data.position.x,
            position_y=node_data.position.y,
            width=node_data.size.width,
            height=node_data.size.height,
            properties=node_data.properties,
            visual_style=node_data.visual_style
        )
        
        db.add(node)
        
        # Update design timestamp
        design.updated_at = datetime.now()
        
        await db.commit()
        await db.refresh(node)
        
        return NodeResponse(**node.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create node: {str(e)}")

@router.get("/{design_id}/nodes", response_model=List[NodeResponse])
async def get_design_nodes(
    design_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user)
):
    """Get all nodes for a design"""
    try:
        # Verify design exists
        design_query = select(Design).where(Design.id == design_id)
        design_result = await db.execute(design_query)
        design = design_result.scalar_one_or_none()
        
        if not design:
            raise HTTPException(status_code=404, detail="Design not found")
        
        # Get nodes
        nodes_query = select(DesignNode).where(DesignNode.design_id == design_id)
        nodes_result = await db.execute(nodes_query)
        nodes = nodes_result.scalars().all()
        
        return [NodeResponse(**node.to_dict()) for node in nodes]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get nodes: {str(e)}")

@router.put("/{design_id}/nodes/{node_id}", response_model=NodeResponse)
async def update_node(
    design_id: UUID,
    node_id: str,
    node_data: NodeUpdate,
    db: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user)
):
    """Update a specific node"""
    try:
        # Get node
        query = select(DesignNode).where(
            DesignNode.design_id == design_id,
            DesignNode.node_id == node_id
        )
        result = await db.execute(query)
        node = result.scalar_one_or_none()
        
        if not node:
            raise HTTPException(status_code=404, detail="Node not found")
        
        # Update fields
        update_data = node_data.dict(exclude_unset=True)
        
        if "position" in update_data:
            node.position_x = update_data["position"]["x"]
            node.position_y = update_data["position"]["y"]
        
        if "size" in update_data:
            node.width = update_data["size"]["width"]
            node.height = update_data["size"]["height"]
        
        for field, value in update_data.items():
            if field not in ["position", "size"] and hasattr(node, field):
                setattr(node, field, value)
        
        # Update design timestamp
        design_query = select(Design).where(Design.id == design_id)
        design_result = await db.execute(design_query)
        design = design_result.scalar_one()
        design.updated_at = datetime.now()
        
        await db.commit()
        await db.refresh(node)
        
        return NodeResponse(**node.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update node: {str(e)}")

@router.delete("/{design_id}/nodes/{node_id}", response_model=BaseResponse)
async def delete_node(
    design_id: UUID,
    node_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user)
):
    """Delete a node and its connections"""
    try:
        # Get node
        query = select(DesignNode).where(
            DesignNode.design_id == design_id,
            DesignNode.node_id == node_id
        )
        result = await db.execute(query)
        node = result.scalar_one_or_none()
        
        if not node:
            raise HTTPException(status_code=404, detail="Node not found")
        
        # Delete connections involving this node
        conn_delete_query = delete(DesignConnection).where(
            DesignConnection.design_id == design_id,
            (DesignConnection.from_node_id == node_id) | 
            (DesignConnection.to_node_id == node_id)
        )
        await db.execute(conn_delete_query)
        
        # Delete node
        await db.delete(node)
        
        # Update design timestamp
        design_query = select(Design).where(Design.id == design_id)
        design_result = await db.execute(design_query)
        design = design_result.scalar_one()
        design.updated_at = datetime.now()
        
        await db.commit()
        
        return BaseResponse(message="Node deleted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete node: {str(e)}")

# ============================================================================
# CONNECTION OPERATIONS
# ============================================================================

@router.post("/{design_id}/connections", response_model=ConnectionResponse, status_code=201)
async def create_connection(
    design_id: UUID,
    connection_data: ConnectionCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user)
):
    """Create a connection between nodes"""
    try:
        # Verify design exists
        design_query = select(Design).where(Design.id == design_id)
        design_result = await db.execute(design_query)
        design = design_result.scalar_one_or_none()
        
        if not design:
            raise HTTPException(status_code=404, detail="Design not found")
        
        # Verify both nodes exist
        from_node_query = select(DesignNode).where(
            DesignNode.design_id == design_id,
            DesignNode.node_id == connection_data.from_node_id
        )
        from_node_result = await db.execute(from_node_query)
        if not from_node_result.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Source node not found")
        
        to_node_query = select(DesignNode).where(
            DesignNode.design_id == design_id,
            DesignNode.node_id == connection_data.to_node_id
        )
        to_node_result = await db.execute(to_node_query)
        if not to_node_result.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Target node not found")
        
        # Check if connection_id already exists in this design
        existing_query = select(DesignConnection).where(
            DesignConnection.design_id == design_id,
            DesignConnection.connection_id == connection_data.connection_id
        )
        existing_result = await db.execute(existing_query)
        if existing_result.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Connection ID already exists in this design")
        
        # Create connection
        connection = DesignConnection(
            design_id=design_id,
            connection_id=connection_data.connection_id,
            from_node_id=connection_data.from_node_id,
            to_node_id=connection_data.to_node_id,
            from_port=connection_data.from_port,
            to_port=connection_data.to_port,
            connection_type=connection_data.connection_type,
            properties=connection_data.properties,
            visual_style=connection_data.visual_style
        )
        
        db.add(connection)
        
        # Update design timestamp
        design.updated_at = datetime.now()
        
        await db.commit()
        await db.refresh(connection)
        
        return ConnectionResponse(**connection.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create connection: {str(e)}")

@router.get("/{design_id}/connections", response_model=List[ConnectionResponse])
async def get_design_connections(
    design_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user)
):
    """Get all connections for a design"""
    try:
        # Verify design exists
        design_query = select(Design).where(Design.id == design_id)
        design_result = await db.execute(design_query)
        design = design_result.scalar_one_or_none()
        
        if not design:
            raise HTTPException(status_code=404, detail="Design not found")
        
        # Get connections
        connections_query = select(DesignConnection).where(DesignConnection.design_id == design_id)
        connections_result = await db.execute(connections_query)
        connections = connections_result.scalars().all()
        
        return [ConnectionResponse(**conn.to_dict()) for conn in connections]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get connections: {str(e)}")

@router.delete("/{design_id}/connections/{connection_id}", response_model=BaseResponse)
async def delete_connection(
    design_id: UUID,
    connection_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: dict = Depends(get_current_user)
):
    """Delete a connection"""
    try:
        # Get connection
        query = select(DesignConnection).where(
            DesignConnection.design_id == design_id,
            DesignConnection.connection_id == connection_id
        )
        result = await db.execute(query)
        connection = result.scalar_one_or_none()
        
        if not connection:
            raise HTTPException(status_code=404, detail="Connection not found")
        
        # Delete connection
        await db.delete(connection)
        
        # Update design timestamp
        design_query = select(Design).where(Design.id == design_id)
        design_result = await db.execute(design_query)
        design = design_result.scalar_one()
        design.updated_at = datetime.now()
        
        await db.commit()
        
        return BaseResponse(message="Connection deleted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete connection: {str(e)}")
