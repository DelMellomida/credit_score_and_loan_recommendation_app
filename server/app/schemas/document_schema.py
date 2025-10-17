from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime

from fastapi import UploadFile, File
from typing import Optional


from fastapi import Depends

class DocumentUploadRequest:
    def __init__(
        brgyCert: Optional[UploadFile] = File(None),
        eSignaturePersonal: Optional[UploadFile] = File(None),
        payslip: Optional[UploadFile] = File(None),
        companyId: Optional[UploadFile] = File(None),
        proofOfBilling: Optional[UploadFile] = File(None),
        eSignatureCoMaker: Optional[UploadFile] = File(None),
    ):
        self._fields = {
            "brgy_cert": brgyCert,
            "e_signature_personal": eSignaturePersonal,
            "payslip": payslip,
            "company_id": companyId,
            "proof_of_billing": proofOfBilling,
            "e_signature_comaker": eSignatureCoMaker,
        }

    def to_dict(self):
        return self._fields

class DocumentResponse(BaseModel):
    document_id: str
    application_id: str
    brgy_cert_url: Optional[str]
    e_signature_personal_url: Optional[str]
    payslip_url: Optional[str]
    company_id_url: Optional[str]
    proof_of_billing_url: Optional[str]
    e_signature_comaker_url: Optional[str]
    file_metadata: Optional[Dict[str, Dict[str, str]]]
    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str