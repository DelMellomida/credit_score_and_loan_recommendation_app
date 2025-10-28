import React, { useState } from 'react';
import type { ReactElement } from 'react';
import { Upload, ArrowLeft, X, FileText, Trash2, ChevronLeft, ChevronRight } from 'lucide-react';
import { PositionField } from './forms/PositionField';
import { validatePosition, Position } from '@/lib/positionValidation';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { ScrollArea } from './ui/scroll-area';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';
import { Applicant } from './ApplicantList';
import { FormData } from '@/lib/formTypes';
import { transformToApplicant } from '../lib/transformData';
import { useAuth } from '../context/AuthContext';
import { deleteLoanApplication } from '@/lib/api';

import { useEffect } from 'react';
import { getApplicationDocuments, getLoanApplication, refreshApplicationDocumentUrls, updateApplicationDocumentUrls } from '@/lib/api';

interface ApplicantOverviewProps {
  applicant: Applicant & { 
    documents?: Record<string, string | null>;
    id: string;
  };
  onBack: () => void;
  onSave: (updatedData: any) => void;
}

interface DocumentUpload {
  id: string;
  name: string;
  file: File | null;
  preview: string | null;
}

interface LoanRecommendation {
  product_name: string;
  max_loanable_amount: number;
  term_in_months: number;
  interest_rate_monthly: number;
  estimated_amortization_per_cutoff: number;
  suitability_score: number;
  is_top_recommendation?: boolean;
}

interface BinaryId {
  $binary: {
    base64: string;
    subType: string;
  };
}

interface MongoId {
  $oid: string;
}

interface ApplicationData {
  _id?: string | MongoId;
  application_id?: string | BinaryId;
  prediction_result?: {
    loan_recommendation?: LoanRecommendation[];
    status?: string;
    final_credit_score?: number;
    probability_of_default?: number;
  };
  model_input_data?: {
    Employment_Tenure_Months?: number;
    Salary_Frequency?: string;
    Housing_Status?: string;
    Years_at_Current_Address?: number;
    Number_of_Dependents?: number;
  };
  ai_explanation?: {
    technical_explanation: string;
    business_explanation: string;
    risk_factors: string;
    recommendations: string;
  };
  timestamp: string;
  status: string;
  loan_officer_id: string;
  applicant_info: any;
  comaker_info: any;
  documents?: Record<string, string | null>;
}

export function ApplicantOverview({
  applicant,
  onBack,
  onSave,
}: ApplicantOverviewProps) {
  // Helper function to calculate risk level and class
  const calculateRisk = (score: number | undefined) => {
    const creditScore = score ?? 0;
    return {
      class: creditScore >= 740 ? 'bg-green-100 text-green-700' :
             creditScore >= 670 ? 'bg-blue-100 text-blue-700' :
             creditScore >= 580 ? 'bg-yellow-100 text-yellow-700' :
             'bg-red-100 text-red-700',
      level: creditScore >= 740 ? 'Very Low Risk' :
             creditScore >= 670 ? 'Low Risk' :
             creditScore >= 580 ? 'Moderate Risk' :
             'High Risk'
    };
  };
  console.log('Applicant Data:', {
    fullApplicant: applicant,
    formData: applicant.formData,
    modelInputData: applicant.formData.other,
    paluwagan: applicant.formData.other.paluwagaParticipation
  });
  const { user } = useAuth();
  const [applicationData, setApplicationData] = useState<ApplicationData | null>(null);
  const [editableData, setEditableData] = useState<FormData>(applicant.formData);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  // Accept both camelCase and snake_case initial values from passed applicant.documents
  const [profilePhoto, setProfilePhoto] = useState<string | null>(
    applicant.documents?.profilePhoto || (applicant.documents?.profile_photo_url as string) || (applicant.documents?.profile_photo as string) || null
  );
  const [idPhoto, setIdPhoto] = useState<string | null>(
    applicant.documents?.idPhoto || (applicant.documents?.valid_id_url as string) || (applicant.documents?.valid_id as string) || null
  );
  
  // Document uploads and previews
  const [documents, setDocuments] = useState({
    brgyCert: null as File | null,
    eSignaturePersonal: null as File | null,
    payslip: null as File | null,
    companyId: null as File | null,
    proofOfBilling: null as File | null,
    eSignatureCoMaker: null as File | null,
  });
  const [documentPreviews, setDocumentPreviews] = useState({
    brgyCert: null as string | null,
    eSignaturePersonal: null as string | null,
    payslip: null as string | null,
    companyId: null as string | null,
    proofOfBilling: null as string | null,
    eSignatureCoMaker: null as string | null,
  });

  // application UUID (from applicationData.application_id) used to refresh signed URLs
  const [applicationUUID, setApplicationUUID] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<'recommendation' | 'prediction' | 'explanation'>('recommendation');

  // Track image retry attempts to avoid infinite refresh loops for all document types
  const [imageRetries, setImageRetries] = useState<Record<string, number>>({
    profilePhoto: 0,
    idPhoto: 0,
    brgyCert: 0,
    eSignaturePersonal: 0,
    payslip: 0,
    companyId: 0,
    proofOfBilling: 0,
    eSignatureCoMaker: 0,
  });
  const MAX_IMAGE_RETRIES = 1; // attempt one automatic refresh per image

  // Fetch application data and document URLs when component mounts
  // Periodically check URLs and refresh if needed (every 4 minutes)
  useEffect(() => {
    const checkUrls = async () => {
      const anyExpiringSoon = [
        profilePhoto,
        idPhoto,
        ...Object.values(documentPreviews)
      ].some(url => url && isUrlExpiringSoon(url));

      if (anyExpiringSoon) {
        console.log('Some URLs are expiring soon, refreshing...');
        await refreshSignedUrls();
      }
    };

    const interval = setInterval(checkUrls, 4 * 60 * 1000);
    return () => clearInterval(interval);
  }, [applicationUUID, user?.token, profilePhoto, idPhoto, documentPreviews]);

  useEffect(() => {
    const fetchApplicationData = async () => {
      try {
        if (!user?.token) {
          throw new Error('No authentication token available');
        }

        // First get the complete loan application using MongoDB _id
        console.log('Fetching complete application data for:', applicant.id);
        const response = await getLoanApplication(applicant.id, user.token);
        console.log('Received application data:', response);
        setApplicationData(response);  // Store the application data in state

        // Extract the application_id (UUID) from the application details
        const applicationId = response?.application_id 
          ? typeof response.application_id === 'string'
            ? response.application_id
            : response.application_id.$binary?.base64 || ''
          : '';

        if (!applicationId) {
          console.warn('No application_id found in application data, cannot fetch documents.');
        } else {
          // store the application UUID for later refresh attempts
          setApplicationUUID(applicationId);
          // Then get the document URLs using the application_id (UUID)
          console.log('Fetching document URLs for application_id:', applicationId);
          const documentResponse = await getApplicationDocuments(applicationId, user.token) as Record<string, string | null>;
          console.log('Received document URLs:', documentResponse);

          if (documentResponse) {
            // Set document previews
            const previews = { ...documentPreviews };
            Object.entries(documentResponse).forEach(([key, value]) => {
              if (value && typeof value === 'string') {
                console.log(`Setting preview for ${key}:`, value.substring(0, 100));
                // Map snake_case keys (and variants) to camelCase
                const profileKeys = ['profile_photo_url', 'profile_photo', 'profilePhoto'];
                const idKeys = ['valid_id_url', 'valid_id', 'idPhoto'];

                if (profileKeys.includes(key)) {
                  setProfilePhoto(value);
                } else if (idKeys.includes(key)) {
                  setIdPhoto(value);
                } else {
                  // Map other document keys and any variants
                  const mappings: Record<string, keyof typeof documentPreviews> = {
                    'brgy_cert_url': 'brgyCert',
                    'brgy_cert': 'brgyCert',
                    'e_signature_personal_url': 'eSignaturePersonal',
                    'e_signature_personal': 'eSignaturePersonal',
                    'payslip_url': 'payslip',
                    'payslip': 'payslip',
                    'company_id_url': 'companyId',
                    'company_id': 'companyId',
                    'proof_of_billing_url': 'proofOfBilling',
                    'proof_of_billing': 'proofOfBilling',
                    'e_signature_comaker_url': 'eSignatureCoMaker',
                    'e_signature_comaker': 'eSignatureCoMaker',
                  };
                  const previewKey = mappings[key];
                  if (previewKey) {
                    previews[previewKey] = value;
                  }
                }
              }
            });
            console.log('Updated previews:', previews);
            setDocumentPreviews(previews);
          }
        }

        // Update the form data with the latest application data
        if (response) {
          // Cast to any since we know the structure is compatible but types don't fully match
          const transformedData = transformToApplicant(response as any);
          setEditableData(transformedData.formData);
        }

      } catch (error) {
        console.error('Error fetching application data:', error);
      }
    };

    fetchApplicationData();
  }, [applicant.id, user?.token]);

  // Utility function to check if URL is about to expire
  const isUrlExpiringSoon = (url: string) => {
    try {
      const urlObj = new URL(url);
      const expiryParam = urlObj.searchParams.get('X-Amz-Expires');
      const dateParam = urlObj.searchParams.get('X-Amz-Date');
      
      if (!expiryParam || !dateParam) return true;
      
      const expirySeconds = parseInt(expiryParam, 10);
      const dateStr = dateParam;
      
      // AWS signature format: YYYYMMDDTHHMMSSZ
      const signatureDate = new Date(
        Date.UTC(
          parseInt(dateStr.slice(0, 4), 10),
          parseInt(dateStr.slice(4, 6), 10) - 1,
          parseInt(dateStr.slice(6, 8), 10),
          parseInt(dateStr.slice(9, 11), 10),
          parseInt(dateStr.slice(11, 13), 10),
          parseInt(dateStr.slice(13, 15), 10)
        )
      );
      
      const expiryTime = signatureDate.getTime() + (expirySeconds * 1000);
      const timeToExpiry = expiryTime - Date.now();
      
      // Return true if URL will expire in less than 5 minutes
      return timeToExpiry < 5 * 60 * 1000;
    } catch (error) {
      console.error('Error parsing URL expiration:', error);
      return true;
    }
  };

  // Helper to refresh signed URLs by calling the dedicated refresh endpoint
  const refreshSignedUrls = async (field?: 'profilePhoto' | 'idPhoto') => {
    if (!applicationUUID || !user?.token) return null;
    try {
      // Check if URLs are expiring soon or already expired
      const needsRefresh = field 
        ? field === 'profilePhoto' 
          ? !profilePhoto || isUrlExpiringSoon(profilePhoto)
          : !idPhoto || isUrlExpiringSoon(idPhoto)
        : Object.values(documentPreviews).some(url => url && isUrlExpiringSoon(url));

      if (!needsRefresh) {
        console.log('URLs are still valid, no refresh needed');
        return null;
      }

      // If specific field requested, only refresh that type
      const documentTypes = field 
        ? [field === 'profilePhoto' ? 'profile_photo' : 'valid_id']
        : undefined;
      
      const refreshed = await refreshApplicationDocumentUrls(
        applicationUUID,
        documentTypes,
        user.token
      );
      if (!refreshed) return null;

      // Update profile/id specifically if present
      if (field === 'profilePhoto') {
        const profile = refreshed.profile_photo_url || refreshed.profile_photo || refreshed.profilePhoto;
        if (profile) setProfilePhoto(profile);
      } else if (field === 'idPhoto') {
        const id = refreshed.valid_id_url || refreshed.valid_id || refreshed.validId || refreshed.idPhoto;
        if (id) setIdPhoto(id);
      } else {
        // General update of all previews
        const previews = { ...documentPreviews };
        const mappings: Record<string, keyof typeof documentPreviews> = {
          'brgy_cert_url': 'brgyCert', 'brgy_cert': 'brgyCert',
          'e_signature_personal_url': 'eSignaturePersonal', 'e_signature_personal': 'eSignaturePersonal',
          'payslip_url': 'payslip', 'payslip': 'payslip',
          'company_id_url': 'companyId', 'company_id': 'companyId',
          'proof_of_billing_url': 'proofOfBilling', 'proof_of_billing': 'proofOfBilling',
          'e_signature_comaker_url': 'eSignatureCoMaker', 'e_signature_comaker': 'eSignatureCoMaker'
        };
        Object.entries(refreshed).forEach(([k, v]) => {
          if (!v) return;
          if (['profile_photo_url','profile_photo','profilePhoto'].includes(k)) setProfilePhoto(v);
          else if (['valid_id_url','valid_id','validId','idPhoto'].includes(k)) setIdPhoto(v);
          else if (mappings[k]) previews[mappings[k]] = v;
        });
        setDocumentPreviews(previews);

        // Update the database with new URLs
        try {
          // Filter out null values and convert to Record<string, string>
          const validUrls = Object.fromEntries(
            Object.entries(refreshed)
              .filter(([_, value]) => value !== null)
          ) as Record<string, string>;
          
          await updateApplicationDocumentUrls(applicationUUID, validUrls, user.token);
          console.log('Successfully updated document URLs in database');
        } catch (error) {
          console.error('Failed to update document URLs in database:', error);
        }
      }
      return refreshed;
    } catch (e) {
      console.error('Failed to refresh signed URLs:', e);
      return null;
    }
  };

  const handleInputChange = (section: keyof FormData, field: string, value: string) => {
    setEditableData(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [field]: value
      }
    }));
  };

  const handleFileUpload = (type: 'profile' | 'id' | keyof typeof documents, e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      console.log('File upload preview generated:', { type, fileName: file.name });
      
      if (type === 'profile') {
        setProfilePhoto(result);
      } else if (type === 'id') {
        setIdPhoto(result);
      } else {
        setDocuments(prev => ({ ...prev, [type]: file }));
        setDocumentPreviews(prev => ({ ...prev, [type]: result }));
      }
    };
    reader.readAsDataURL(file);
  };

  const handleSave = () => {
    onSave({
      formData: editableData,
      documents,
      profilePhoto,
      idPhoto,
    });
  };

  // Get loan recommendations and prediction data from application data
  const predictionResult = applicationData?.prediction_result;
  const recommendations = predictionResult?.loan_recommendation || [];
  const topRecommendation = recommendations.find(rec => rec.is_top_recommendation) || recommendations[0];
  const hasAiExplanation = Boolean(applicationData?.ai_explanation);
  
  const recommendedLoan = topRecommendation ? {
    product: topRecommendation.product_name,
    amount: `₱${topRecommendation.max_loanable_amount.toLocaleString()}`,
    term: `${topRecommendation.term_in_months} months`,
    interest: `${topRecommendation.interest_rate_monthly}% monthly`,
    amortization: `₱${topRecommendation.estimated_amortization_per_cutoff.toLocaleString()} per cutoff`,
    suitability: `${topRecommendation.suitability_score}% match`,
  } : {
    product: 'No recommendation available',
    amount: '₱0',
    term: 'N/A',
    interest: 'N/A',
    amortization: 'N/A',
    suitability: 'N/A',
  };

  return (
    <div className="h-full flex flex-col gap-4 overflow-hidden">
      {/* Image Preview Modal */}
      {previewImage && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
          onClick={() => setPreviewImage(null)}
        >
          <div className="relative max-w-4xl max-h-[90vh] w-full h-full flex items-center justify-center">
            <Button
              variant="ghost"
              size="icon"
              className="absolute top-2 right-2 bg-white/10 hover:bg-white/20 text-white"
              onClick={() => setPreviewImage(null)}
            >
              <X className="h-4 w-4" />
            </Button>
            <img
              src={previewImage}
              alt="Full size preview"
              className="max-w-full max-h-full object-contain"
              onClick={(e) => e.stopPropagation()}
            />
          </div>
        </div>
      )}
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            onClick={onBack}
            className="flex items-center gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
          <h2 className="text-2xl">Applicant Overview</h2>
        </div>
        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button
              className="flex items-center gap-2 bg-red-600 hover:bg-red-700"
            >
              <Trash2 className="h-4 w-4" />
              Delete Application
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Delete Loan Application</AlertDialogTitle>
              <AlertDialogDescription>
                Are you sure you want to delete this loan application? This action will permanently delete all related documents and cannot be undone.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction
                onClick={async () => {
                  if (!user?.token || !applicationUUID) return;
                  try {
                    await deleteLoanApplication(applicationUUID, user.token);
                    // Navigate back to list view
                    onBack();
                  } catch (error) {
                    console.error('Error deleting application:', error);
                  }
                }}
                className="bg-red-600 hover:bg-red-700"
              >
                Delete
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>

      {/* Main Content */}
      <div className="flex gap-6 flex-1 overflow-hidden">
        {/* Left Side - Borrower Details */}
        <div className="flex-1 overflow-auto">
          <Card>
            <CardHeader>
              <CardTitle>Borrower Information</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Photo Uploads */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>1x1 Photo</Label>
                  <div className="mt-2 border-2 border-dashed rounded-lg p-4 text-center">
                    {profilePhoto ? (
                      <div className="relative w-32 h-32 mx-auto">
                        <img 
                          src={profilePhoto} 
                          alt="Profile" 
                          className="w-full h-full object-cover rounded cursor-pointer"
                          onClick={() => setPreviewImage(profilePhoto)}
                          onError={async (e) => {
                            console.error('Failed to load profile photo:', profilePhoto);
                            const retries = imageRetries.profilePhoto || 0;
                            if (retries < MAX_IMAGE_RETRIES) {
                              setImageRetries(prev => ({ ...prev, profilePhoto: retries + 1 }));
                              const refreshed = await refreshSignedUrls('profilePhoto');
                              if (!refreshed) {
                                e.currentTarget.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="8" r="4"/><path d="M20 21a8 8 0 1 0-16 0"/></svg>';
                              }
                            } else {
                              e.currentTarget.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="8" r="4"/><path d="M20 21a8 8 0 1 0-16 0"/></svg>';
                            }
                          }}
                        />
                      </div>
                    ) : (
                      <div className="w-32 h-32 mx-auto bg-gray-100 rounded flex items-center justify-center">
                        <Upload className="h-8 w-8 text-gray-400" />
                      </div>
                    )}
                    <input
                      type="file"
                      id="profile-photo"
                      accept="image/*"
                      className="hidden"
                      onChange={(e) => handleFileUpload('profile', e)}
                    />
                    <Button
                      variant="outline"
                      size="sm"
                      className="mt-2"
                      onClick={() => document.getElementById('profile-photo')?.click()}
                    >
                      {profilePhoto ? 'Change Photo' : 'Upload Photo'}
                    </Button>
                  </div>
                </div>
                
                <div>
                  <Label>ID Photo</Label>
                  <div className="mt-2 border-2 border-dashed rounded-lg p-4 text-center">
                    {idPhoto ? (
                      <div className="relative w-32 h-32 mx-auto">
                        <img 
                          src={idPhoto} 
                          alt="ID" 
                          className="w-full h-full object-cover rounded cursor-pointer"
                          onClick={() => setPreviewImage(idPhoto)}
                          onError={async (e) => {
                            console.error('Failed to load ID photo:', idPhoto);
                            const retries = imageRetries.idPhoto || 0;
                            if (retries < MAX_IMAGE_RETRIES) {
                              setImageRetries(prev => ({ ...prev, idPhoto: retries + 1 }));
                              const refreshed = await refreshSignedUrls('idPhoto');
                              if (!refreshed) {
                                e.currentTarget.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 8h10"/><path d="M7 12h10"/><path d="M7 16h10"/></svg>';
                              }
                            } else {
                              e.currentTarget.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 8h10"/><path d="M7 12h10"/><path d="M7 16h10"/></svg>';
                            }
                          }}
                        />
                      </div>
                    ) : (
                      <div className="w-32 h-32 mx-auto bg-gray-100 rounded flex items-center justify-center">
                        <Upload className="h-8 w-8 text-gray-400" />
                      </div>
                    )}
                    <input
                      type="file"
                      id="id-photo"
                      accept="image/*"
                      className="hidden"
                      onChange={(e) => handleFileUpload('id', e)}
                    />
                    <Button
                      variant="outline"
                      size="sm"
                      className="mt-2"
                      onClick={() => document.getElementById('id-photo')?.click()}
                    >
                      {idPhoto ? 'Change ID Photo' : 'Upload ID Photo'}
                    </Button>
                  </div>
                </div>
              </div>

              {/* Personal Data */}
              <div className="space-y-4">
                <h3 className="text-lg border-b pb-2">Personal Data</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label>Full Name</Label>
                    <Input
                      value={editableData.personal.fullName}
                      onChange={(e) => handleInputChange('personal', 'fullName', e.target.value)}
                    />
                  </div>
                  <div>
                    <Label>Contact Number</Label>
                    <Input
                      value={editableData.personal.contactNo}
                      onChange={(e) => handleInputChange('personal', 'contactNo', e.target.value)}
                    />
                  </div>
                  <div className="col-span-2">
                    <Label>Address</Label>
                    <Input
                      value={editableData.personal.address}
                      onChange={(e) => handleInputChange('personal', 'address', e.target.value)}
                    />
                  </div>
                  <div>
                    <Label>Head of Household</Label>
                    <Input
                      value={editableData.personal.headOfHousehold}
                      onChange={(e) => handleInputChange('personal', 'headOfHousehold', e.target.value)}
                    />
                  </div>
                  <div>
                    <Label>Dependents</Label>
                    <Input
                      value={editableData.personal.dependents}
                      onChange={(e) => handleInputChange('personal', 'dependents', e.target.value)}
                    />
                  </div>
                  <div>
                    <Label>Years Living Here</Label>
                    <Input
                      value={editableData.personal.yearsLivingHere}
                      onChange={(e) => handleInputChange('personal', 'yearsLivingHere', e.target.value)}
                    />
                  </div>
                  <div>
                    <Label>Housing Status</Label>
                    <Input
                      value={editableData.personal.housingStatus}
                      onChange={(e) => handleInputChange('personal', 'housingStatus', e.target.value)}
                    />
                  </div>
                </div>
              </div>

              {/* Employee Data */}
              <div className="space-y-4">
                <h3 className="text-lg border-b pb-2">Employee Data</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label>Company Name</Label>
                    <Input
                      value={editableData.employee.companyName}
                      onChange={(e) => handleInputChange('employee', 'companyName', e.target.value)}
                    />
                  </div>
                  <div>
                    <Label>Sector</Label>
                    <Input
                      value={editableData.employee.sector}
                      onChange={(e) => handleInputChange('employee', 'sector', e.target.value)}
                    />
                  </div>
                  <PositionField
                    value={validatePosition(editableData.employee.position)}
                    onChange={(value) => handleInputChange('employee', 'position', value)}
                  />
                  <div>
                    <Label>Employment Duration</Label>
                    <Input
                      value={editableData.employee.employmentDuration}
                      onChange={(e) => handleInputChange('employee', 'employmentDuration', e.target.value)}
                    />
                  </div>
                  <div>
                    <Label>Salary</Label>
                    <Input
                      value={editableData.employee.salary}
                      onChange={(e) => handleInputChange('employee', 'salary', e.target.value)}
                    />
                  </div>
                  <div>
                    <Label>Type of Salary</Label>
                    <Input
                      value={editableData.employee.typeOfSalary}
                      onChange={(e) => handleInputChange('employee', 'typeOfSalary', e.target.value)}
                    />
                  </div>
                </div>
              </div>

              {/* Other Data */}
              <div className="space-y-4">
                <h3 className="text-lg border-b pb-2">Other Data</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label>Community Position</Label>
                    <Input
                      value={editableData.other.communityPosition}
                      onChange={(e) => handleInputChange('other', 'communityPosition', e.target.value)}
                    />
                  </div>
                  <div>
                    <Label>Paluwaga Participation</Label>
                    <Input
                      value={editableData.other.paluwagaParticipation}
                      onChange={(e) => handleInputChange('other', 'paluwagaParticipation', e.target.value)}
                    />
                  </div>
                  <div className="col-span-2">
                    <Label>Other Income Sources</Label>
                    <Input
                      value={editableData.other.otherIncomeSources}
                      onChange={(e) => handleInputChange('other', 'otherIncomeSources', e.target.value)}
                    />
                  </div>
                  <div className="col-span-2">
                    <Label>Disaster Preparedness Strategy</Label>
                    <Input
                      value={editableData.other.disasterPreparednessStrategy}
                      onChange={(e) => handleInputChange('other', 'disasterPreparednessStrategy', e.target.value)}
                    />
                  </div>
                </div>
              </div>

              {/* Co-Maker Data */}
              <div className="space-y-4">
                <h3 className="text-lg border-b pb-2">Co-Maker Data</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label>Full Name</Label>
                    <Input
                      value={editableData.coMaker.fullName}
                      onChange={(e) => handleInputChange('coMaker', 'fullName', e.target.value)}
                    />
                  </div>
                  <div>
                    <Label>Contact Number</Label>
                    <Input
                      value={editableData.coMaker.contactNo}
                      onChange={(e) => handleInputChange('coMaker', 'contactNo', e.target.value)}
                    />
                  </div>
                  <div className="col-span-2">
                    <Label>Address</Label>
                    <Input
                      value={editableData.coMaker.address}
                      onChange={(e) => handleInputChange('coMaker', 'address', e.target.value)}
                    />
                  </div>
                  <div>
                    <Label>How Many Months/Years</Label>
                    <Input
                      value={editableData.coMaker.howManyMonthsYears}
                      onChange={(e) => handleInputChange('coMaker', 'howManyMonthsYears', e.target.value)}
                    />
                  </div>
                  <div>
                    <Label>Salary</Label>
                    <Input
                      value={editableData.coMaker.salary}
                      onChange={(e) => handleInputChange('coMaker', 'salary', e.target.value)}
                    />
                  </div>
                  <div className="col-span-2">
                    <Label>Relationship with Applicant</Label>
                    <Input
                      value={editableData.coMaker.relationshipWithApplicant}
                      onChange={(e) => handleInputChange('coMaker', 'relationshipWithApplicant', e.target.value)}
                    />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Side - Summary & Documents (Scrollable) */}
        <div className="w-96">
          <ScrollArea className="h-full">
            <div className="space-y-4 pr-4">
              {/* Summary Carousel */}
              <Card className="relative">
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-gray-800 flex-1">
                    {applicationData?.prediction_result?.status === "Approved" ? (
                      <span className="text-green-600">Approved</span>
                    ) : applicationData?.prediction_result?.status === "Denied" ? (
                      <span className="text-red-600">Denied</span>
                    ) : (
                      <span className="text-yellow-600">Pending</span>
                    )}
                  </CardTitle>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => {
                        setActiveSection(prev => 
                          prev === 'explanation' ? 'prediction' :
                          prev === 'prediction' ? 'recommendation' :
                          'explanation'
                        );
                      }}
                      className="h-8 w-8"
                    >
                      <ChevronLeft className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => {
                        setActiveSection(prev => 
                          prev === 'recommendation' ? 'prediction' :
                          prev === 'prediction' ? 'explanation' :
                          'recommendation'
                        );
                      }}
                      className="h-8 w-8"
                    >
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                  </div>
                </CardHeader>

                <CardContent className="space-y-2">
                  {activeSection === 'recommendation' && (
                    <div className="space-y-2">
                      <h3 className="font-semibold">Loan Recommendation</h3>
                      <div>
                        <p className="text-sm text-gray-600">Recommended Product</p>
                        <p className="font-medium">{recommendedLoan.product}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Maximum Loanable Amount</p>
                        <p className="font-medium">{recommendedLoan.amount}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Term</p>
                        <p className="font-medium">{recommendedLoan.term}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Interest Rate</p>
                        <p className="font-medium">{recommendedLoan.interest}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Estimated Amortization</p>
                        <p className="font-medium">{recommendedLoan.amortization}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Suitability Score</p>
                        <p className="font-medium text-green-700">{recommendedLoan.suitability}</p>
                      </div>
                    </div>
                  )}

                  {activeSection === 'prediction' && (
                    <div className="space-y-2">
                      <h3 className="font-semibold">Credit Score Analysis</h3>
                      <div>
                        <p className="text-sm text-gray-600">Credit Score</p>
                        <p className="font-medium">
                          {applicationData?.prediction_result?.final_credit_score}
                          {(() => {
                            const creditScore = applicationData?.prediction_result?.final_credit_score ?? 0;
                            const risk = calculateRisk(creditScore);
                            return (
                              <span className={`ml-2 px-2 py-0.5 text-xs rounded ${risk.class}`}>
                                {risk.level}
                              </span>
                            );
                          })()}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Default Probability</p>
                        <p className="font-medium">
                          {((applicationData?.prediction_result?.probability_of_default ?? 0) * 100).toFixed(2)}%
                        </p>
                      </div>
                      {/* Key risk factors section removed */}
                    </div>
                  )}

                  {activeSection === 'explanation' && applicationData?.ai_explanation && (
                    <div className="space-y-2">
                      <h3 className="font-semibold">AI Analysis</h3>
                      {applicationData.ai_explanation.technical_explanation.includes("Error") ? (
                        <div className="text-yellow-600 text-sm">
                          AI explanation is currently unavailable for this application.
                        </div>
                      ) : (
                        <>
                          <div>
                            <p className="text-sm font-medium">Business Analysis</p>
                            <p className="text-sm text-gray-600 mt-1">{applicationData.ai_explanation.business_explanation}</p>
                          </div>
                          <div>
                            <p className="text-sm font-medium">Risk Factors</p>
                            <p className="text-sm text-gray-600 mt-1">{applicationData.ai_explanation.risk_factors}</p>
                          </div>
                          <div>
                            <p className="text-sm font-medium">Recommendations</p>
                            <p className="text-sm text-gray-600 mt-1">{applicationData.ai_explanation.recommendations}</p>
                          </div>
                        </>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Document Gallery */}
              <Card>
                <CardHeader>
                  <CardTitle>Documents</CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Personal Documents */}
                  <div className="space-y-3">
                    <h4 className="text-sm text-gray-600">Personal</h4>
                    <DocumentUploadField
                      label="Barangay Certificate"
                      id="brgy-cert"
                      file={documents.brgyCert}
                      preview={documentPreviews.brgyCert}
                      onChange={(e) => handleFileUpload('brgyCert', e)}
                      applicationUUID={applicationUUID}
                      user={user}
                      imageRetries={imageRetries}
                      setImageRetries={setImageRetries}
                      documentPreviews={documentPreviews}
                      setDocumentPreviews={setDocumentPreviews}
                      setProfilePhoto={setProfilePhoto}
                      setIdPhoto={setIdPhoto}
                      setPreviewImage={setPreviewImage}
                    />
                    <DocumentUploadField
                      label="E-Signature"
                      id="e-signature-personal"
                      file={documents.eSignaturePersonal}
                      preview={documentPreviews.eSignaturePersonal}
                      onChange={(e) => handleFileUpload('eSignaturePersonal', e)}
                      applicationUUID={applicationUUID}
                      user={user}
                      imageRetries={imageRetries}
                      setImageRetries={setImageRetries}
                      documentPreviews={documentPreviews}
                      setDocumentPreviews={setDocumentPreviews}
                      setProfilePhoto={setProfilePhoto}
                      setIdPhoto={setIdPhoto}
                      setPreviewImage={setPreviewImage}
                    />
                  </div>

                  {/* Employee Documents */}
                  <div className="space-y-3">
                    <h4 className="text-sm text-gray-600">Employee</h4>
                    <DocumentUploadField
                      label="Payslip"
                      id="payslip"
                      file={documents.payslip}
                      preview={documentPreviews.payslip}
                      onChange={(e) => handleFileUpload('payslip', e)}
                      applicationUUID={applicationUUID}
                      user={user}
                      imageRetries={imageRetries}
                      setImageRetries={setImageRetries}
                      documentPreviews={documentPreviews}
                      setDocumentPreviews={setDocumentPreviews}
                      setProfilePhoto={setProfilePhoto}
                      setIdPhoto={setIdPhoto}
                      setPreviewImage={setPreviewImage}
                    />
                    <DocumentUploadField
                      label="Company ID"
                      id="company-id"
                      file={documents.companyId}
                      preview={documentPreviews.companyId}
                      onChange={(e) => handleFileUpload('companyId', e)}
                      applicationUUID={applicationUUID}
                      user={user}
                      imageRetries={imageRetries}
                      setImageRetries={setImageRetries}
                      documentPreviews={documentPreviews}
                      setDocumentPreviews={setDocumentPreviews}
                      setProfilePhoto={setProfilePhoto}
                      setIdPhoto={setIdPhoto}
                      setPreviewImage={setPreviewImage}
                    />
                  </div>

                  {/* Other Documents */}
                  <div className="space-y-3">
                    <h4 className="text-sm text-gray-600">Other</h4>
                    <DocumentUploadField
                      label="Proof of Billing"
                      id="proof-billing"
                      file={documents.proofOfBilling}
                      preview={documentPreviews.proofOfBilling}
                      onChange={(e) => handleFileUpload('proofOfBilling', e)}
                      applicationUUID={applicationUUID}
                      user={user}
                      imageRetries={imageRetries}
                      setImageRetries={setImageRetries}
                      documentPreviews={documentPreviews}
                      setDocumentPreviews={setDocumentPreviews}
                      setProfilePhoto={setProfilePhoto}
                      setIdPhoto={setIdPhoto}
                      setPreviewImage={setPreviewImage}
                    />
                  </div>

                  {/* Co-Maker Documents */}
                  <div className="space-y-3">
                    <h4 className="text-sm text-gray-600">Co-Maker</h4>
                    <DocumentUploadField
                      label="E-Signature"
                      id="e-signature-comaker"
                      file={documents.eSignatureCoMaker}
                      preview={documentPreviews.eSignatureCoMaker}
                      onChange={(e) => handleFileUpload('eSignatureCoMaker', e)}
                      applicationUUID={applicationUUID}
                      user={user}
                      imageRetries={imageRetries}
                      setImageRetries={setImageRetries}
                      documentPreviews={documentPreviews}
                      setDocumentPreviews={setDocumentPreviews}
                      setProfilePhoto={setProfilePhoto}
                      setIdPhoto={setIdPhoto}
                      setPreviewImage={setPreviewImage}
                    />
                  </div>
                </CardContent>
              </Card>
            </div>
          </ScrollArea>
        </div>
      </div>
    </div>
  );
}

// Helper component for document upload fields
interface DocumentUploadFieldProps {
  label: string;
  id: string;
  file: File | null;
  preview?: string | null;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  applicationUUID: string | null;
  user: { token: string } | null;
  imageRetries: Record<string, number>;
  setImageRetries: (value: React.SetStateAction<Record<string, number>>) => void;
  documentPreviews: {
    brgyCert: string | null;
    eSignaturePersonal: string | null;
    payslip: string | null;
    companyId: string | null;
    proofOfBilling: string | null;
    eSignatureCoMaker: string | null;
  };
  setDocumentPreviews: (value: React.SetStateAction<{
    brgyCert: string | null;
    eSignaturePersonal: string | null;
    payslip: string | null;
    companyId: string | null;
    proofOfBilling: string | null;
    eSignatureCoMaker: string | null;
  }>) => void;
  setProfilePhoto: (value: React.SetStateAction<string | null>) => void;
  setIdPhoto: (value: React.SetStateAction<string | null>) => void;
  setPreviewImage: (value: string | null) => void;
  MAX_IMAGE_RETRIES?: number;
}

const DocumentUploadField: React.FC<DocumentUploadFieldProps> = ({
  label,
  id,
  file,
  preview,
  onChange,
  applicationUUID,
  user,
  imageRetries,
  setImageRetries,
  documentPreviews,
  setDocumentPreviews,
  setProfilePhoto,
  setIdPhoto,
  setPreviewImage,
  MAX_IMAGE_RETRIES = 1
}) => {
  return (
    <div className="border rounded-lg p-3">
      <Label className="text-sm">{label}</Label>
      <div className="mt-2 border-2 border-dashed rounded-lg p-4">
        {/* Preview section */}
        <div className="mb-4">
          {preview ? (
            preview.toLowerCase().endsWith('.pdf') ? (
              <div className="w-full h-32 bg-gray-100 rounded flex items-center justify-center">
                <div className="text-center">
                  <FileText className="h-12 w-12 text-blue-500 mx-auto mb-2" />
                  <span className="text-sm text-gray-600">PDF Document</span>
                </div>
              </div>
            ) : (
              <div className="relative w-full h-32 mx-auto">
                <img 
                  src={preview} 
                  alt={`${label} preview`} 
                  className="w-full h-full object-contain rounded cursor-pointer"
                  onClick={() => setPreviewImage(preview)}
                  onError={async (e) => {
                    console.error(`Failed to load ${label}:`, preview);
                    const fieldKey = id.replace(/-/g, '') as keyof typeof imageRetries;
                    const retries = imageRetries[fieldKey] || 0;

                    if (retries < MAX_IMAGE_RETRIES && applicationUUID && user?.token) {
                      try {
                        setImageRetries(prev => ({ ...prev, [fieldKey]: retries + 1 }));
                        // Refresh all document URLs when any image fails
                        const refreshed = await refreshApplicationDocumentUrls(applicationUUID, undefined, user.token);
                        
                        if (refreshed) {
                          // Update all document previews with fresh URLs
                          const newPreviews = { ...documentPreviews };
                          const mappings: Record<string, keyof typeof documentPreviews> = {
                            'brgy_cert_url': 'brgyCert',
                            'e_signature_personal_url': 'eSignaturePersonal',
                            'payslip_url': 'payslip',
                            'company_id_url': 'companyId',
                            'proof_of_billing_url': 'proofOfBilling',
                            'e_signature_comaker_url': 'eSignatureCoMaker'
                          };

                          Object.entries(refreshed).forEach(([key, value]) => {
                            if (!value) return;
                            if (key === 'profile_photo_url') setProfilePhoto(value);
                            else if (key === 'valid_id_url') setIdPhoto(value);
                            else {
                              const previewKey = mappings[key];
                              if (previewKey) newPreviews[previewKey] = value;
                            }
                          });

                          setDocumentPreviews(newPreviews);
                          return; // Exit early if refresh successful
                        }
                      } catch (error) {
                        console.error('Failed to refresh URLs:', error);
                      }
                    }

                    // Show fallback if refresh failed or max retries reached
                    if (e.currentTarget.parentElement) {
                      e.currentTarget.parentElement.innerHTML = `
                        <div class="w-full h-full bg-gray-100 rounded flex items-center justify-center">
                          <div class="text-center">
                            <svg class="h-12 w-12 text-gray-400 mx-auto mb-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v1m6 11h2m-6 0h-2m0 0H8m13 0a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            <span class="text-sm text-gray-500">Failed to load image</span>
                          </div>
                        </div>
                      `;
                    }
                  }}
                />
              </div>
            )
          ) : (
            <div className="w-full h-32 bg-gray-100 rounded flex items-center justify-center">
              <div className="text-center">
                <Upload className="h-12 w-12 text-gray-400 mx-auto mb-2" />
                <span className="text-sm text-gray-500">No file uploaded</span>
              </div>
            </div>
          )}
        </div>
        
        {/* Upload controls */}
        <div className="flex items-center justify-between gap-2">
          <span className="text-sm flex-1 truncate">
            {file ? (
              <span className="text-green-600">{file.name}</span>
            ) : preview ? (
              <span className="text-blue-600">File uploaded</span>
            ) : (
              <span className="text-gray-400">Choose a file...</span>
            )}
          </span>
          <input
            type="file"
            id={id}
            accept="image/*,.pdf"
            className="hidden"
            onChange={onChange}
          />
          <Button
            variant="outline"
            size="sm"
            onClick={() => document.getElementById(id)?.click()}
            className="flex items-center gap-2"
          >
            <Upload className="h-4 w-4" />
            {file || preview ? 'Change File' : 'Upload File'}
          </Button>
        </div>
      </div>
    </div>
  );
}
