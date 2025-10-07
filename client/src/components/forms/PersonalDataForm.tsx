import React from 'react';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { PersonalData } from '../../app/page';

interface PersonalDataFormProps {
  data: PersonalData;
  updateData: (data: Partial<PersonalData>) => void;
}

export function PersonalDataForm({ data, updateData }: PersonalDataFormProps) {
  const handleInputChange = (field: keyof PersonalData, value: string) => {
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
            placeholder="Enter full name"
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
          <Label htmlFor="headOfHousehold">Head of Household</Label>
          <Select onValueChange={(value) => handleInputChange('headOfHousehold', value)}>
            <SelectTrigger>
              <SelectValue placeholder="Select head of household" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="self">Self</SelectItem>
              <SelectItem value="spouse">Spouse</SelectItem>
              <SelectItem value="parent">Parent</SelectItem>
              <SelectItem value="other">Other</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="dependents">Dependents</Label>
          <Select onValueChange={(value) => handleInputChange('dependents', value)}>
            <SelectTrigger>
              <SelectValue placeholder="Number of dependents" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="0">0</SelectItem>
              <SelectItem value="1">1</SelectItem>
              <SelectItem value="2">2</SelectItem>
              <SelectItem value="3">3</SelectItem>
              <SelectItem value="4">4</SelectItem>
              <SelectItem value="5+">5+</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="yearsLivingHere">Years of Living Here</Label>
          <Input
            id="yearsLivingHere"
            value={data.yearsLivingHere}
            onChange={(e) => handleInputChange('yearsLivingHere', e.target.value)}
            placeholder="Enter number of years"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="housingStatus">Housing Status</Label>
          <Select onValueChange={(value) => handleInputChange('housingStatus', value)}>
            <SelectTrigger>
              <SelectValue placeholder="Select housing status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="owned">Owned</SelectItem>
              <SelectItem value="rented">Rented</SelectItem>
              <SelectItem value="family-owned">Family Owned</SelectItem>
              <SelectItem value="other">Other</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
    </div>
  );
}