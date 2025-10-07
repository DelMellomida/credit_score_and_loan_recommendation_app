// Utility to transform frontend FormData to backend FullLoanApplicationRequest
import { FormData } from '../app/page';

export function transformLoanFormData(formData: FormData) {
  // Map enums and fields carefully
  function mapEmploymentSector(sector: string) {
    if (sector === 'Public') return 'Public';
    if (sector === 'Private') return 'Private';
    return 'Private';
  }
  function mapSalaryFrequency(type: string) {
    switch (type) {
      case 'Monthly': return 'Monthly';
      case 'Bimonthly': return 'Bimonthly';
      case 'Biweekly': return 'Biweekly';
      case 'Weekly': return 'Weekly';
      default: return 'Monthly';
    }
  }
  function mapHousingStatus(status: string) {
    if (status === 'Owned') return 'Owned';
    if (status === 'Rented') return 'Rented';
    return 'Owned';
  }
  function mapYesNo(val: string) {
    return val === 'Yes' ? 'Yes' : 'No';
  }
  function mapComakerRelationship(rel: string) {
    switch (rel) {
      case 'Spouse': return 'Spouse';
      case 'Sibling': return 'Sibling';
      case 'Parent': return 'Parent';
      case 'Friend': return 'Friend';
      default: return 'Friend';
    }
  }
  function mapCommunityRole(role: string) {
    switch (role) {
      case 'None': return 'None';
      case 'Member': return 'Member';
      case 'Leader': return 'Leader';
      case 'Multiple Leader': return 'Multiple Leader';
      default: return 'None';
    }
  }
  function mapPaluwaganParticipation(val: string) {
    switch (val) {
      case 'Never': return 'Never';
      case 'Rarely': return 'Rarely';
      case 'Sometimes': return 'Sometimes';
      case 'Frequently': return 'Frequently';
      default: return 'Never';
    }
  }
  function mapOtherIncomeSource(val: string) {
    switch (val) {
      case 'None': return 'None';
      case 'OFW Remittance': return 'OFW Remittance';
      case 'Freelance': return 'Freelance';
      case 'Business': return 'Business';
      default: return 'None';
    }
  }
  function mapDisasterPreparedness(val: string) {
    switch (val) {
      case 'None': return 'None';
      case 'Savings': return 'Savings';
      case 'Insurance': return 'Insurance';
      case 'Community Plan': return 'Community Plan';
      default: return 'None';
    }
  }

  // Transform
  return {
    applicant_info: {
      full_name: formData.personal.fullName,
      contact_number: formData.personal.contactNo,
      address: formData.personal.address,
      salary: formData.employee.salary,
      job: formData.employee.position,
    },
    comaker_info: {
      full_name: formData.coMaker.fullName,
      contact_number: formData.coMaker.contactNo,
    },
    model_input_data: {
      Employment_Sector: mapEmploymentSector(formData.employee.sector),
      Employment_Tenure_Months: parseInt(formData.employee.employmentDuration) || 0,
      Net_Salary_Per_Cutoff: parseFloat(formData.employee.salary) || 0,
      Salary_Frequency: mapSalaryFrequency(formData.employee.typeOfSalary),
      Housing_Status: mapHousingStatus(formData.personal.housingStatus),
      Years_at_Current_Address: parseFloat(formData.personal.yearsLivingHere) || 0,
      Household_Head: mapYesNo(formData.personal.headOfHousehold),
      Number_of_Dependents: parseInt(formData.personal.dependents) || 0,
      Comaker_Relationship: mapComakerRelationship(formData.coMaker.relationshipWithApplicant),
      Comaker_Employment_Tenure_Months: parseInt(formData.coMaker.howManyMonthsYears) || 0,
      Comaker_Net_Salary_Per_Cutoff: parseFloat(formData.coMaker.salary) || 0,
      Has_Community_Role: mapCommunityRole(formData.other.communityPosition),
      Paluwagan_Participation: mapPaluwaganParticipation(formData.other.paluwagaParticipation),
      Other_Income_Source: mapOtherIncomeSource(formData.other.otherIncomeSources),
      Disaster_Preparedness: mapDisasterPreparedness(formData.other.disasterPreparednessStrategy),
      Is_Renewing_Client: 0, // Set as needed
      Grace_Period_Usage_Rate: 0.0, // Set as needed
      Late_Payment_Count: 0, // Set as needed
      Had_Special_Consideration: 0, // Set as needed
    },
  };
}
