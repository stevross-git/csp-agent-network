# backend/api/endpoints/designs.py
"""
Design Management API Endpoints with Enhanced RBAC
=================================================
FastAPI endpoints for visual design CRUD operations with role-based access control
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, delete, update, func, and_, or_
from typing import List, Optional, Dict, Any
from uuid import UUID
import uuid
from datetime import datetime, timedelta

# Import models and schemas
from backend.models.database_models import Design, DesignNode, DesignConnection, ExecutionSession, AuditLog
from backend.schemas.api_schemas import (
    DesignCreate, DesignUpdate, DesignResponse, DesignListResponse,
    NodeCreate, NodeUpdate, NodeResponse,
    ConnectionCreate, ConnectionUpdate, ConnectionResponse,
    ExecutionConfig, ExecutionResponse, BaseResponse, ErrorResponse,
    AuditLogEntry
)
from backend.database.connection import get_db_session

# NEW - Import RBAC system
from backend.auth.rbac import (
    get_current_user_unified, UnifiedUserInfo, require_designer, require_authenticated,
    require_design_create, require_design_delete, RBACService, Permission, UserRole
)

# Create router
router = APIRouter(prefix="/api/designs", tags=["designs"])

# NEW - Helper function for audit logging
async def log_design_activity(
    action: str,
    user: UnifiedUserInfo,
    design_id: str = None,
    success: bool = True,
    details: Dict[str, Any] = None,
    db_session: AsyncSession = None
):
    """Log design-related activity for audit trail"""
    try:
        if db_session:
            audit_log = AuditLog(
                user_id=user.user_id,
                user_email=user.email,
                action=action,
                resource_type="design",
                resource_id=design_id,
                details=details or {},
                success=success
            )
            db_session.add(audit_log)
            # Note: commit is handled by the calling function
    except Exception as e:
        print(f"Failed to log audit event: {e}")

# NEW - Helper function to check design ownership/permissions
async def check_design_access(
    design_id: UUID, 
    user: UnifiedUserInfo, 
    permission: Permission,
    db_session: AsyncSession
) -> Design:
    """Check if user has access to design and return design object"""
    
    # Get design
    query = select(Design).where(Design.id == design_id)
    result = await db_session.execute(query)
    design = result.scalar_one_or_none()
    
    if not design:
        raise HTTPException(status_code=404, detail="Design not found")
    
    # Check permissions
    rbac = RBACService()
    
    # Admins can access everything
    if user.has_role(UserRole.ADMIN) or user.has_role(UserRole.SUPER_ADMIN):
        return design
    
    # Check if user owns the design
    user_owns_design = (
        (design.created_by and str(design.created_by) == user.user_id) or
        (design.created_by_local_user_id and str(design.created_by_local_user_id) == user.user_id)
    )
    
    # For read operations, check if design is public or user owns it
    if permission == Permission.READ_DESIGN:
        if design.is_public or user_owns_design:
            return design
    
    # For write operations, user must own the design or have admin permissions
    elif permission in [Permission.UPDATE_DESIGN, Permission.DELETE_DESIGN, Permission.EXECUTE_DESIGN]:
        if user_owns_design:
            return design
    
    # Check role-based permissions as fallback
    if user.has_permission(permission):
        return design
    
    # Access denied
    raise HTTPException(
        status_code=403,
        detail=f"Access denied. Required permission: {permission.value}"
    )

# ============================================================================
# DESIGN CRUD OPERATIONS (Enhanced with RBAC)
# ============================================================================

@router.post("/", response_model=DesignResponse, status_code=201)
async def create_design(
    design_data: DesignCreate,
    current_user: UnifiedUserInfo = Depends(require_design_create),  # NEW - RBAC enforcement
    db: AsyncSession = Depends(get_db_session),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Create a new visual design"""
    try:
        # Create design with proper user association
        design = Design(
            name=design_data.name,
            description=design_data.description,
            version=design_data.version,
            canvas_settings=design_data.canvas_settings,
            metadata=design_data.metadata,
            is_public=design_data.is_public,
            is_template=design_data.is_template,
            tags=design_data.tags
        )
        
        # Set creator based on auth method
        if current_user.auth_method == 'azure':
            design.created_by = UUID(current_user.user_id)
        else:  # local auth
            design.created_by_local_user_id = UUID(current_user.user_id)
        
        db.add(design)
        await db.commit()
        await db.refresh(design)
        
        # NEW - Log activity
        background_tasks.add_task(
            log_design_activity,
            action="create_design",
            user=current_user,
            design_id=str(design.id),
            success=True,
            details={"name": design.name, "is_public": design.is_public},
            db_session=db
        )
        
        return DesignResponse(**design.to_dict())
        
    except Exception as e:
        await db.rollback()
        
        # NEW - Log failure
        background_tasks.add_task(
            log_design_activity,
            action="create_design",
            user=current_user,
            success=False,
            details={"error": str(e), "name": design_data.name},
            db_session=db
        )
        
        raise HTTPException(status_code=500, detail=f"Failed to create design: {str(e)}")

@router.get("/", response_model=DesignListResponse)
async def list_designs(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search in name and description"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    is_public: Optional[bool] = Query(None, description="Filter by public/private"),
    is_template: Optional[bool] = Query(None, description="Filter by template status"),
    current_user: UnifiedUserInfo = Depends(require_authenticated),  # NEW - Require auth
    db: AsyncSession = Depends(get_db_session)
):
    """List designs with enhanced filtering and permissions"""
    try:
        # Build base query
        query = select(Design)
        
        # NEW - Apply access control filters
        access_filters = []
        
        # Public designs are visible to everyone
        access_filters.append(Design.is_public == True)
        
        # User's own designs
        if current_user.auth_method == 'azure':
            access_filters.append(Design.created_by == UUID(current_user.user_id))
        else:
            access_filters.append(Design.created_by_local_user_id == UUID(current_user.user_id))
        
        # Admins can see all designs
        if current_user.has_role(UserRole.ADMIN) or current_user.has_role(UserRole.SUPER_ADMIN):
            # Remove access filters for admins
            pass
        else:
            query = query.where(or_(*access_filters))
        
        # Apply filters
        if search:
            search_filter = or_(
                Design.name.ilike(f"%{search}%"),
                Design.description.ilike(f"%{search}%")
            )
            query = query.where(search_filter)
        
        if tags:
            # Filter by any of the provided tags
            tag_filters = [Design.tags.contains([tag]) for tag in tags]
            query = query.where(or_(*tag_filters))
        
        if is_public is not None:
            query = query.where(Design.is_public == is_public)
        
        if is_template is not None:
            query = query.where(Design.is_template == is_template)
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination and ordering
        query = query.order_by(Design.updated_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        # Execute query
        result = await db.execute(query)
        designs = result.scalars().all()
        
        return DesignListResponse(
            designs=[DesignResponse(**design.to_dict()) for design in designs],
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list designs: {str(e)}")

@router.get("/{design_id}", response_model=DesignResponse)
async def get_design(
    design_id: UUID = Path(..., description="Design ID"),
    current_user: UnifiedUserInfo = Depends(require_authenticated),  # NEW - Require auth
    db: AsyncSession = Depends(get_db_session)
):
    """Get a specific design with permission check"""
    try:
        # NEW - Check access permissions
        design = await check_design_access(design_id, current_user, Permission.READ_DESIGN, db)
        
        # Load relationships
        query = select(Design).options(
            selectinload(Design.nodes),
            selectinload(Design.connections)
        ).where(Design.id == design_id)
        
        result = await db.execute(query)
        design_with_relations = result.scalar_one()
        
        return DesignResponse(**design_with_relations.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get design: {str(e)}")

@router.put("/{design_id}", response_model=DesignResponse)
async def update_design(
    design_id: UUID = Path(..., description="Design ID"),
    design_data: DesignUpdate = ...,
    current_user: UnifiedUserInfo = Depends(require_authenticated),  # NEW - Auth required
    db: AsyncSession = Depends(get_db_session),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Update a design with permission check"""
    try:
        # NEW - Check access permissions
        design = await check_design_access(design_id, current_user, Permission.UPDATE_DESIGN, db)
        
        # Track changes for audit log
        changes = {}
        
        # Update fields
        update_data = design_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            if hasattr(design, field):
                old_value = getattr(design, field)
                if old_value != value:
                    changes[field] = {"old": old_value, "new": value}
                    setattr(design, field, value)
        
        design.updated_at = datetime.now()
        
        await db.commit()
        await db.refresh(design)
        
        # NEW - Log activity
        if changes:
            background_tasks.add_task(
                log_design_activity,
                action="update_design",
                user=current_user,
                design_id=str(design.id),
                success=True,
                details={"changes": changes},
                db_session=db
            )
        
        return DesignResponse(**design.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        
        # NEW - Log failure
        background_tasks.add_task(
            log_design_activity,
            action="update_design",
            user=current_user,
            design_id=str(design_id),
            success=False,
            details={"error": str(e)},
            db_session=db
        )
        
        raise HTTPException(status_code=500, detail=f"Failed to update design: {str(e)}")

@router.delete("/{design_id}", response_model=BaseResponse)
async def delete_design(
    design_id: UUID = Path(..., description="Design ID"),
    current_user: UnifiedUserInfo = Depends(require_design_delete),  # NEW - RBAC enforcement
    db: AsyncSession = Depends(get_db_session),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Delete a design with permission check"""
    try:
        # NEW - Check access permissions
        design = await check_design_access(design_id, current_user, Permission.DELETE_DESIGN, db)
        
        design_name = design.name  # Store for logging
        
        # Delete design (cascade will handle nodes and connections)
        await db.delete(design)
        await db.commit()
        
        # NEW - Log activity
        background_tasks.add_task(
            log_design_activity,
            action="delete_design",
            user=current_user,
            design_id=str(design_id),
            success=True,
            details={"name": design_name},
            db_session=db
        )
        
        return BaseResponse(message="Design deleted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        
        # NEW - Log failure
        background_tasks.add_task(
            log_design_activity,
            action="delete_design",
            user=current_user,
            design_id=str(design_id),
            success=False,
            details={"error": str(e)},
            db_session=db
        )
        
        raise HTTPException(status_code=500, detail=f"Failed to delete design: {str(e)}")

# ============================================================================
# NODE OPERATIONS (Enhanced with RBAC)
# ============================================================================

@router.post("/{design_id}/nodes", response_model=NodeResponse, status_code=201)
async def create_node(
    design_id: UUID = Path(..., description="Design ID"),
    node_data: NodeCreate = ...,
    current_user: UnifiedUserInfo = Depends(require_authenticated),  # NEW - Auth required
    db: AsyncSession = Depends(get_db_session),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Create a node in a design with permission check"""
    try:
        # NEW - Check access permissions
        design = await check_design_access(design_id, current_user, Permission.UPDATE_DESIGN, db)
        
        # Check if node ID already exists in design
        existing_node_query = select(DesignNode).where(
            DesignNode.design_id == design_id,
            DesignNode.node_id == node_data.node_id
        )
        existing_result = await db.execute(existing_node_query)
        if existing_result.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Node ID already exists in design")
        
        # Create node
        node = DesignNode(
            design_id=design_id,
            node_id=node_data.node_id,
            component_type=node_data.component_type,
            component_config=node_data.component_config,
            position_x=node_data.position["x"],
            position_y=node_data.position["y"],
            width=node_data.size.get("width", 200),
            height=node_data.size.get("height", 100),
            metadata=node_data.metadata
        )
        
        db.add(node)
        
        # Update design timestamp
        design.updated_at = datetime.now()
        
        await db.commit()
        await db.refresh(node)
        
        # NEW - Log activity
        background_tasks.add_task(
            log_design_activity,
            action="create_node",
            user=current_user,
            design_id=str(design_id),
            success=True,
            details={
                "node_id": node_data.node_id,
                "component_type": node_data.component_type
            },
            db_session=db
        )
        
        return NodeResponse(**node.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        
        # NEW - Log failure
        background_tasks.add_task(
            log_design_activity,
            action="create_node",
            user=current_user,
            design_id=str(design_id),
            success=False,
            details={"error": str(e), "node_id": node_data.node_id},
            db_session=db
        )
        
        raise HTTPException(status_code=500, detail=f"Failed to create node: {str(e)}")

@router.get("/{design_id}/nodes", response_model=List[NodeResponse])
async def get_design_nodes(
    design_id: UUID = Path(..., description="Design ID"),
    current_user: UnifiedUserInfo = Depends(require_authenticated),  # NEW - Auth required
    db: AsyncSession = Depends(get_db_session)
):
    """Get all nodes for a design with permission check"""
    try:
        # NEW - Check access permissions
        await check_design_access(design_id, current_user, Permission.READ_DESIGN, db)
        
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
    design_id: UUID = Path(..., description="Design ID"),
    node_id: str = Path(..., description="Node ID"),
    node_data: NodeUpdate = ...,
    current_user: UnifiedUserInfo = Depends(require_authenticated),  # NEW - Auth required
    db: AsyncSession = Depends(get_db_session),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Update a specific node with permission check"""
    try:
        # NEW - Check access permissions
        await check_design_access(design_id, current_user, Permission.UPDATE_DESIGN, db)
        
        # Get node
        query = select(DesignNode).where(
            DesignNode.design_id == design_id,
            DesignNode.node_id == node_id
        )
        result = await db.execute(query)
        node = result.scalar_one_or_none()
        
        if not node:
            raise HTTPException(status_code=404, detail="Node not found")
        
        # Track changes
        changes = {}
        
        # Update fields
        update_data = node_data.dict(exclude_unset=True)
        
        if "position" in update_data:
            old_pos = {"x": node.position_x, "y": node.position_y}
            new_pos = update_data["position"]
            if old_pos != new_pos:
                changes["position"] = {"old": old_pos, "new": new_pos}
                node.position_x = new_pos["x"]
                node.position_y = new_pos["y"]
        
        if "size" in update_data:
            old_size = {"width": node.width, "height": node.height}
            new_size = update_data["size"]
            if old_size != new_size:
                changes["size"] = {"old": old_size, "new": new_size}
                node.width = new_size["width"]
                node.height = new_size["height"]
        
        for field, value in update_data.items():
            if field not in ["position", "size"] and hasattr(node, field):
                old_value = getattr(node, field)
                if old_value != value:
                    changes[field] = {"old": old_value, "new": value}
                    setattr(node, field, value)
        
        # Update timestamps
        node.updated_at = datetime.now()
        
        # Update design timestamp
        design_query = select(Design).where(Design.id == design_id)
        design_result = await db.execute(design_query)
        design = design_result.scalar_one()
        design.updated_at = datetime.now()
        
        await db.commit()
        await db.refresh(node)
        
        # NEW - Log activity
        if changes:
            background_tasks.add_task(
                log_design_activity,
                action="update_node",
                user=current_user,
                design_id=str(design_id),
                success=True,
                details={"node_id": node_id, "changes": changes},
                db_session=db
            )
        
        return NodeResponse(**node.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        
        # NEW - Log failure
        background_tasks.add_task(
            log_design_activity,
            action="update_node",
            user=current_user,
            design_id=str(design_id),
            success=False,
            details={"error": str(e), "node_id": node_id},
            db_session=db
        )
        
        raise HTTPException(status_code=500, detail=f"Failed to update node: {str(e)}")

@router.delete("/{design_id}/nodes/{node_id}", response_model=BaseResponse)
async def delete_node(
    design_id: UUID = Path(..., description="Design ID"),
    node_id: str = Path(..., description="Node ID"),
    current_user: UnifiedUserInfo = Depends(require_authenticated),  # NEW - Auth required
    db: AsyncSession = Depends(get_db_session),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Delete a node and its connections with permission check"""
    try:
        # NEW - Check access permissions
        await check_design_access(design_id, current_user, Permission.UPDATE_DESIGN, db)
        
        # Get node
        query = select(DesignNode).where(
            DesignNode.design_id == design_id,
            DesignNode.node_id == node_id
        )
        result = await db.execute(query)
        node = result.scalar_one_or_none()
        
        if not node:
            raise HTTPException(status_code=404, detail="Node not found")
        
        node_component_type = node.component_type  # Store for logging
        
        # Delete connections involving this node
        conn_delete_query = delete(DesignConnection).where(
            DesignConnection.design_id == design_id,
            or_(
                DesignConnection.from_node_id == node_id,
                DesignConnection.to_node_id == node_id
            )
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
        
        # NEW - Log activity
        background_tasks.add_task(
            log_design_activity,
            action="delete_node",
            user=current_user,
            design_id=str(design_id),
            success=True,
            details={
                "node_id": node_id,
                "component_type": node_component_type
            },
            db_session=db
        )
        
        return BaseResponse(message="Node deleted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        
        # NEW - Log failure
        background_tasks.add_task(
            log_design_activity,
            action="delete_node",
            user=current_user,
            design_id=str(design_id),
            success=False,
            details={"error": str(e), "node_id": node_id},
            db_session=db
        )
        
        raise HTTPException(status_code=500, detail=f"Failed to delete node: {str(e)}")

# ============================================================================
# CONNECTION OPERATIONS (Enhanced with RBAC)
# ============================================================================

@router.post("/{design_id}/connections", response_model=ConnectionResponse, status_code=201)
async def create_connection(
    design_id: UUID = Path(..., description="Design ID"),
    connection_data: ConnectionCreate = ...,
    current_user: UnifiedUserInfo = Depends(require_authenticated),  # NEW - Auth required
    db: AsyncSession = Depends(get_db_session),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Create a connection between nodes with permission check"""
    try:
        # NEW - Check access permissions
        design = await check_design_access(design_id, current_user, Permission.UPDATE_DESIGN, db)
        
        # Verify both nodes exist
        from_node_query = select(DesignNode).where(
            DesignNode.design_id == design_id,
            DesignNode.node_id == connection_data.from_node_id
        )
        from_node_result = await db.execute(from_node_query)
        from_node = from_node_result.scalar_one_or_none()
        
        if not from_node:
            raise HTTPException(status_code=400, detail="Source node not found")
        
        to_node_query = select(DesignNode).where(
            DesignNode.design_id == design_id,
            DesignNode.node_id == connection_data.to_node_id
        )
        to_node_result = await db.execute(to_node_query)
        to_node = to_node_result.scalar_one_or_none()
        
        if not to_node:
            raise HTTPException(status_code=400, detail="Target node not found")
        
        # Check if connection ID already exists
        existing_conn_query = select(DesignConnection).where(
            DesignConnection.design_id == design_id,
            DesignConnection.connection_id == connection_data.connection_id
        )
        existing_result = await db.execute(existing_conn_query)
        if existing_result.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Connection ID already exists")
        
        # Create connection
        connection = DesignConnection(
            design_id=design_id,
            connection_id=connection_data.connection_id,
            from_node_id=connection_data.from_node_id,
            from_port=connection_data.from_port,
            to_node_id=connection_data.to_node_id,
            to_port=connection_data.to_port,
            connection_type=connection_data.connection_type,
            style=connection_data.style,
            metadata=connection_data.metadata
        )
        
        db.add(connection)
        
        # Update design timestamp
        design.updated_at = datetime.now()
        
        await db.commit()
        await db.refresh(connection)
        
        # NEW - Log activity
        background_tasks.add_task(
            log_design_activity,
            action="create_connection",
            user=current_user,
            design_id=str(design_id),
            success=True,
            details={
                "connection_id": connection_data.connection_id,
                "from_node": connection_data.from_node_id,
                "to_node": connection_data.to_node_id
            },
            db_session=db
        )
        
        return ConnectionResponse(**connection.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        
        # NEW - Log failure
        background_tasks.add_task(
            log_design_activity,
            action="create_connection",
            user=current_user,
            design_id=str(design_id),
            success=False,
            details={"error": str(e), "connection_id": connection_data.connection_id},
            db_session=db
        )
        
        raise HTTPException(status_code=500, detail=f"Failed to create connection: {str(e)}")

@router.get("/{design_id}/connections", response_model=List[ConnectionResponse])
async def get_design_connections(
    design_id: UUID = Path(..., description="Design ID"),
    current_user: UnifiedUserInfo = Depends(require_authenticated),  # NEW - Auth required
    db: AsyncSession = Depends(get_db_session)
):
    """Get all connections for a design with permission check"""
    try:
        # NEW - Check access permissions
        await check_design_access(design_id, current_user, Permission.READ_DESIGN, db)
        
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
    design_id: UUID = Path(..., description="Design ID"),
    connection_id: str = Path(..., description="Connection ID"),
    current_user: UnifiedUserInfo = Depends(require_authenticated),  # NEW - Auth required
    db: AsyncSession = Depends(get_db_session),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Delete a connection with permission check"""
    try:
        # NEW - Check access permissions
        await check_design_access(design_id, current_user, Permission.UPDATE_DESIGN, db)
        
        # Get connection
        query = select(DesignConnection).where(
            DesignConnection.design_id == design_id,
            DesignConnection.connection_id == connection_id
        )
        result = await db.execute(query)
        connection = result.scalar_one_or_none()
        
        if not connection:
            raise HTTPException(status_code=404, detail="Connection not found")
        
        connection_details = {
            "from_node": connection.from_node_id,
            "to_node": connection.to_node_id,
            "connection_type": connection.connection_type
        }
        
        # Delete connection
        await db.delete(connection)
        
        # Update design timestamp
        design_query = select(Design).where(Design.id == design_id)
        design_result = await db.execute(design_query)
        design = design_result.scalar_one()
        design.updated_at = datetime.now()
        
        await db.commit()
        
        # NEW - Log activity
        background_tasks.add_task(
            log_design_activity,
            action="delete_connection",
            user=current_user,
            design_id=str(design_id),
            success=True,
            details={
                "connection_id": connection_id,
                **connection_details
            },
            db_session=db
        )
        
        return BaseResponse(message="Connection deleted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        
        # NEW - Log failure
        background_tasks.add_task(
            log_design_activity,
            action="delete_connection",
            user=current_user,
            design_id=str(design_id),
            success=False,
            details={"error": str(e), "connection_id": connection_id},
            db_session=db
        )
        
        raise HTTPException(status_code=500, detail=f"Failed to delete connection: {str(e)}")

# ============================================================================
# NEW - EXECUTION OPERATIONS (Enhanced with RBAC)
# ============================================================================

@router.post("/{design_id}/execute", response_model=ExecutionResponse, status_code=201)
async def execute_design(
    design_id: UUID = Path(..., description="Design ID"),
    config: ExecutionConfig = ...,
    current_user: UnifiedUserInfo = Depends(require_authenticated),  # NEW - Auth required
    db: AsyncSession = Depends(get_db_session),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Execute a design with permission check"""
    try:
        # NEW - Check access permissions
        design = await check_design_access(design_id, current_user, Permission.EXECUTE_DESIGN, db)
        
        # Create execution session
        execution = ExecutionSession(
            design_id=design_id,
            status="pending",
            configuration=config.dict(),
            started_at=datetime.now()
        )
        
        db.add(execution)
        await db.commit()
        await db.refresh(execution)
        
        # NEW - Log activity
        background_tasks.add_task(
            log_design_activity,
            action="execute_design",
            user=current_user,
            design_id=str(design_id),
            success=True,
            details={
                "execution_id": str(execution.id),
                "timeout_seconds": config.timeout_seconds,
                "parallel_execution": config.parallel_execution
            },
            db_session=db
        )
        
        # TODO: Start actual execution in background
        # This would typically involve queuing the execution job
        
        return ExecutionResponse(**execution.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        
        # NEW - Log failure
        background_tasks.add_task(
            log_design_activity,
            action="execute_design",
            user=current_user,
            design_id=str(design_id),
            success=False,
            details={"error": str(e)},
            db_session=db
        )
        
        raise HTTPException(status_code=500, detail=f"Failed to execute design: {str(e)}")

# ============================================================================
# NEW - DESIGN ANALYTICS (Admin/Owner only)
# ============================================================================

@router.get("/{design_id}/analytics", tags=["analytics"])
async def get_design_analytics(
    design_id: UUID = Path(..., description="Design ID"),
    current_user: UnifiedUserInfo = Depends(require_authenticated),
    db: AsyncSession = Depends(get_db_session)
):
    """Get design analytics (owner or admin only)"""
    try:
        # Check access permissions (must own design or be admin)
        design = await check_design_access(design_id, current_user, Permission.READ_DESIGN, db)
        
        # Get execution statistics
        execution_query = select(ExecutionSession).where(ExecutionSession.design_id == design_id)
        executions = await db.execute(execution_query)
        execution_list = executions.scalars().all()
        
        total_executions = len(execution_list)
        successful_executions = len([e for e in execution_list if e.status == "completed"])
        failed_executions = len([e for e in execution_list if e.status == "failed"])
        
        # Calculate average execution time
        completed_executions = [e for e in execution_list if e.ended_at and e.started_at]
        avg_time = 0
        if completed_executions:
            total_time = sum((e.ended_at - e.started_at).total_seconds() for e in completed_executions)
            avg_time = total_time / len(completed_executions)
        
        # Get node count by component type
        nodes_query = select(DesignNode).where(DesignNode.design_id == design_id)
        nodes_result = await db.execute(nodes_query)
        nodes = nodes_result.scalars().all()
        
        component_counts = {}
        for node in nodes:
            comp_type = node.component_type
            component_counts[comp_type] = component_counts.get(comp_type, 0) + 1
        
        return {
            "design_id": str(design_id),
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": (successful_executions / total_executions * 100) if total_executions > 0 else 0,
            "average_execution_time_seconds": avg_time,
            "last_execution": execution_list[-1].created_at.isoformat() if execution_list else None,
            "component_usage": component_counts,
            "total_nodes": len(nodes),
            "total_connections": len(design.connections) if hasattr(design, 'connections') else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")