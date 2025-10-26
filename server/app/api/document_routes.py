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
    profilePhoto: Optional[UploadFile] = File(None),
    validId: Optional[UploadFile] = File(None),
    brgyCert: Optional[UploadFile] = File(None),
    eSignaturePersonal: Optional[UploadFile] = File(None),
    payslip: Optional[UploadFile] = File(None),
    companyId: Optional[UploadFile] = File(None),
    proofOfBilling: Optional[UploadFile] = File(None),
    eSignatureCoMaker: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_active_user)
):
    files = {
        "profile_photo": profilePhoto,
        "valid_id": validId,
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

@router.get("/application/{application_id}")
async def get_documents_by_application(application_id: str, current_user: dict = Depends(get_current_active_user)):
    doc = await service.get_documents_by_application_id(application_id)
    return {
        "brgy_cert_url": doc.brgy_cert_url,
        "e_signature_personal_url": doc.e_signature_personal_url,
        "payslip_url": doc.payslip_url,
        "company_id_url": doc.company_id_url,
        "proof_of_billing_url": doc.proof_of_billing_url,
        "e_signature_comaker_url": doc.e_signature_comaker_url,
        "profile_photo_url": doc.profile_photo_url,
        "valid_id_url": doc.valid_id_url
    }

@router.get("/application/{application_id}/refresh-urls")
async def refresh_application_document_urls(
    application_id: str, 
    document_types: Optional[List[str]] = None,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Refresh signed URLs for an application's documents.
    Optionally specify document types to refresh only specific URLs.
    """
    doc = await service.get_documents_by_application_id(application_id)
    # If document_types specified, only return those URLs
    response = {}
    all_fields = [
        "brgy_cert_url", "e_signature_personal_url", "payslip_url",
        "company_id_url", "proof_of_billing_url", "e_signature_comaker_url",
        "profile_photo_url", "valid_id_url"
    ]
    
    fields_to_refresh = document_types if document_types else all_fields
    
    # Validate requested document types
    if document_types:
        invalid_types = [t for t in document_types if f"{t}_url" not in all_fields]
        if invalid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid document types: {', '.join(invalid_types)}"
            )
    
    # Build response with only requested/available URLs
    for field in fields_to_refresh:
        url_field = f"{field}_url" if not field.endswith("_url") else field
        if hasattr(doc, url_field):
            response[url_field] = getattr(doc, url_field)
    
    return response

@router.put("/{document_id}", response_model=DocumentResponse)
async def update_documents(
    document_id: str,
    profilePhoto: Optional[UploadFile] = File(None),
    validId: Optional[UploadFile] = File(None),
    brgyCert: Optional[UploadFile] = File(None),
    eSignaturePersonal: Optional[UploadFile] = File(None),
    payslip: Optional[UploadFile] = File(None),
    companyId: Optional[UploadFile] = File(None),
    proofOfBilling: Optional[UploadFile] = File(None),
    eSignatureCoMaker: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_active_user)
):
    files = {
        "profile_photo": profilePhoto,
        "valid_id": validId,
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