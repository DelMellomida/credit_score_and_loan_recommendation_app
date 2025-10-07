import React from 'react';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { CoMakerData } from '../../app/page';

interface CoMakerDataFormProps {
  data: CoMakerData;
  updateData: (data: Partial<CoMakerData>) => void;
}

export function CoMakerDataForm({ data, updateData }: CoMakerDataFormProps) {
  const handleInputChange = (field: keyof CoMakerData, value: string) => {
    updateData({ [field]: value });
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="fullName">Full Name</Label>
          <Input
            id="fullName"
            value={data.fullName}
            onChange={(e) => handleInputChange('fullName', e.target.value)}
            placeholder="Enter co-maker's full name"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="contactNo">Contact No</Label>
          <Input
            id="contactNo"
            value={data.contactNo}
            onChange={(e) => handleInputChange('contactNo', e.target.value)}
            placeholder="Enter contact number"
          />
        </div>
      </div>

      <div className="space-y-2">
        <Label htmlFor="address">Address</Label>
        <Input
          id="address"
          value={data.address}
          onChange={(e) => handleInputChange('address', e.target.value)}
          placeholder="Enter complete address"
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="howManyMonthsYears">How many months/years</Label>
          <Input
            id="howManyMonthsYears"
            value={data.howManyMonthsYears}
            onChange={(e) => handleInputChange('howManyMonthsYears', e.target.value)}
            placeholder="Duration of relationship"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="salary">Salary</Label>
          <Input
            id="salary"
            value={data.salary}
            onChange={(e) => handleInputChange('salary', e.target.value)}
            placeholder="Enter monthly salary"
          />
        </div>
      </div>

      <div className="space-y-2">
        <Label htmlFor="relationshipWithApplicant">Relationship with the Applicant</Label>
        <Select onValueChange={(value) => handleInputChange('relationshipWithApplicant', value)}>
          <SelectTrigger>
            <SelectValue placeholder="Select relationship" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="spouse">Spouse</SelectItem>
            <SelectItem value="parent">Parent</SelectItem>
            <SelectItem value="sibling">Sibling</SelectItem>
            <SelectItem value="friend">Friend</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}