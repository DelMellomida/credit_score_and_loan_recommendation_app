from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from typing import Optional, List
from app.services.document_service import DocumentService
from app.schemas.document_schema import DocumentUploadRequest, DocumentResponse
from app.core.auth_dependencies import get_current_active_user

router = APIRouter(prefix="/documents", tags=["Documents"])
service = DocumentService()

@router.post("/", response_model=DocumentResponse)
async def create_documents(
    application_id: str,
    brgyCert: Optional[UploadFile] = File(None),
    eSignaturePersonal: Optional[UploadFile] = File(None),
    payslip: Optional[UploadFile] = File(None),
    companyId: Optional[UploadFile] = File(None),
    proofOfBilling: Optional[UploadFile] = File(None),
    eSignatureCoMaker: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_active_user)
):
    files = {
        "brgy_cert": brgyCert,
        "e_signature_personal": eSignaturePersonal,
        "payslip": payslip,
        "company_id": companyId,
        "proof_of_billing": proofOfBilling,
        "e_signature_comaker": eSignatureCoMaker,
    }
    files = {k: v for k, v in files.items() if v}
    doc = await service.create_documents(application_id, files, current_user["email"])
    return doc

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str, current_user: dict = Depends(get_current_active_user)):
    doc = await service.get_document_by_id(document_id)
    return doc

@router.get("/application/{application_id}", response_model=DocumentResponse)
async def get_documents_by_application(application_id: str, current_user: dict = Depends(get_current_active_user)):
    doc = await service.get_documents_by_application_id(application_id)
    return doc

@router.put("/{document_id}", response_model=DocumentResponse)
async def update_documents(
    document_id: str,
    brgyCert: Optional[UploadFile] = File(None),
    eSignaturePersonal: Optional[UploadFile] = File(None),
    payslip: Optional[UploadFile] = File(None),
    companyId: Optional[UploadFile] = File(None),
    proofOfBilling: Optional[UploadFile] = File(None),
    eSignatureCoMaker: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_active_user)
):
    files = {
        "brgy_cert": brgyCert,
        "e_signature_personal": eSignaturePersonal,
        "payslip": payslip,
        "company_id": companyId,
        "proof_of_billing": proofOfBilling,
        "e_signature_comaker": eSignatureCoMaker,
    }
    files = {k: v for k, v in files.items() if v}
    doc = await service.update_documents(document_id, files, current_user["email"])
    return doc

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_documents(document_id: str, current_user: dict = Depends(get_current_active_user)):
    await service.delete_documents(document_id)
    return {"detail": "Document deleted"}

@router.delete("/{document_id}/{field}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_single_file(document_id: str, field: str, current_user: dict = Depends(get_current_active_user)):
    await service.delete_single_file(document_id, field)
    return {"detail": f"{field} deleted"}

@router.get("/{document_id}/refresh", response_model=DocumentResponse)
async def refresh_signed_urls(document_id: str, current_user: dict = Depends(get_current_active_user)):
    doc = await service.get_document_by_id(document_id)
    return doc