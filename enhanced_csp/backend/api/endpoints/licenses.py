from fastapi import APIRouter, HTTPException, Depends
from typing import List
from uuid import UUID
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.models.database_models import LicenseRecord
from backend.schemas.api_schemas import LicenseCreate, LicenseResponse
from backend.database.connection import get_db_session

router = APIRouter(prefix="/api/licenses", tags=["licenses"])


@router.get("/", response_model=List[LicenseResponse])
async def list_licenses(db: AsyncSession = Depends(get_db_session)):
    result = await db.execute(select(LicenseRecord))
    records = result.scalars().all()
    return [LicenseResponse(**rec.to_dict()) for rec in records]


@router.post("/", response_model=LicenseResponse, status_code=201)
async def create_license(item: LicenseCreate, db: AsyncSession = Depends(get_db_session)):
    record = LicenseRecord(
        product=item.product,
        key=item.key,
        expires_at=item.expires_at,
        active=item.active,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return LicenseResponse(**record.to_dict())


@router.put("/{license_id}", response_model=LicenseResponse)
async def update_license(license_id: str, item: LicenseCreate, db: AsyncSession = Depends(get_db_session)):
    result = await db.execute(select(LicenseRecord).where(LicenseRecord.id == UUID(license_id)))
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=404, detail="License not found")

    record.product = item.product
    record.key = item.key
    record.expires_at = item.expires_at
    record.active = item.active
    record.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(record)
    return LicenseResponse(**record.to_dict())


@router.delete("/{license_id}", status_code=204)
async def delete_license(license_id: str, db: AsyncSession = Depends(get_db_session)):
    result = await db.execute(select(LicenseRecord).where(LicenseRecord.id == UUID(license_id)))
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=404, detail="License not found")

    await db.delete(record)
    await db.commit()
