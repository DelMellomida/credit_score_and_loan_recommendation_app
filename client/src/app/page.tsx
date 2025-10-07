"use client";

import React, { useState } from 'react';
import { LoginForm } from '../components/LoginForm';
import { Header } from '../components/Header';
import { ProcessForm } from '../components/ProcessForm';
import { ResultPreview } from '../components/ResultPreview';
import { useAuth } from '../context/AuthContext';

export interface PersonalData {
  fullName: string;
  contactNo: string;
  address: string;
  headOfHousehold: string;
  dependents: string;
  yearsLivingHere: string;
  housingStatus: string;
}

export interface EmployeeData {
  companyName: string;
  sector: string;
  position: string;
  employmentDuration: string;
  salary: string;
  typeOfSalary: string;
}

export interface OtherData {
  communityPosition: string;
  paluwagaParticipation: string;
  otherIncomeSources: string;
  disasterPreparednessStrategy: string;
}

export interface CoMakerData {
  fullName: string;
  contactNo: string;
  address: string;
  howManyMonthsYears: string;
  salary: string;
  relationshipWithApplicant: string;
}

export interface FormData {
  personal: PersonalData;
  employee: EmployeeData;
  other: OtherData;
  coMaker: CoMakerData;
}
export default function App() {
  const { user, loading, error } = useAuth();
  const [formData, setFormData] = useState<FormData>({
    personal: {
      fullName: '',
      contactNo: '',
      address: '',
      headOfHousehold: '',
      dependents: '',
      yearsLivingHere: '',
      housingStatus: '',
    },
    employee: {
      companyName: '',
      sector: '',
      position: '',
      employmentDuration: '',
      salary: '',
      typeOfSalary: '',
    },
    other: {
      communityPosition: '',
      paluwagaParticipation: '',
      otherIncomeSources: '',
      disasterPreparednessStrategy: '',
    },
    coMaker: {
      fullName: '',
      contactNo: '',
      address: '',
      howManyMonthsYears: '',
      salary: '',
      relationshipWithApplicant: '',
    },
  });
  // Store backend result
  const [loanResult, setLoanResult] = useState<any>(null);

  const updateFormData = (section: keyof FormData, data: any) => {
    setFormData(prev => ({
      ...prev,
      [section]: { ...prev[section], ...data }
    }));
  };

  const newApplicant = () => {
    setFormData({
      personal: {
        fullName: '',
        contactNo: '',
        address: '',
        headOfHousehold: '',
        dependents: '',
        yearsLivingHere: '',
        housingStatus: '',
      },
      employee: {
        companyName: '',
        sector: '',
        position: '',
        employmentDuration: '',
        salary: '',
        typeOfSalary: '',
      },
      other: {
        communityPosition: '',
        paluwagaParticipation: '',
        otherIncomeSources: '',
        disasterPreparednessStrategy: '',
      },
      coMaker: {
        fullName: '',
        contactNo: '',
        address: '',
        howManyMonthsYears: '',
        salary: '',
        relationshipWithApplicant: '',
      },
    });
    setLoanResult(null);
  };

  // Show loading spinner if loading
  if (loading) return <div>Loading...</div>;
  // Show error if error
  if (error) return <div className="text-red-500">{error}</div>;
  // Show login if not authenticated
  if (!user) {
    return <LoginForm />;
  }

  // (Removed old isAuthenticated and LoginFormWrapper logic)

  return (
    <div className="h-screen bg-gray-50 overflow-hidden">
      <Header />
      <div className="flex gap-6 p-6 max-w-7xl mx-auto h-[calc(100vh-80px)] overflow-hidden">
        {/* Process Form Section */}
        <div className="flex-1">
          <ProcessForm
            formData={formData}
            updateFormData={updateFormData}
            newApplicant={newApplicant}
            setLoanResult={setLoanResult}
            token={user.token}
          />
        </div>
        {/* Result Preview Section */}
        <div className="w-96 overflow-hidden">
          <ResultPreview formData={formData} loanResult={loanResult} />
        </div>
      </div>
    </div>
  );
}

// (Removed old LoginFormWrapper and unused code)