import React from 'react';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { EmployeeData } from '../../app/page';

interface EmployeeDataFormProps {
  data: EmployeeData;
  updateData: (data: Partial<EmployeeData>) => void;
}

export function EmployeeDataForm({ data, updateData }: EmployeeDataFormProps) {
  const handleInputChange = (field: keyof EmployeeData, value: string) => {
    updateData({ [field]: value });
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="companyName">Company Name</Label>
          <Input
            id="companyName"
            value={data.companyName}
            onChange={(e) => handleInputChange('companyName', e.target.value)}
            placeholder="Enter company name"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="sector">Sector</Label>
          <Select onValueChange={(value) => handleInputChange('sector', value)}>
            <SelectTrigger>
              <SelectValue placeholder="Select sector" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="public">Public</SelectItem>
              <SelectItem value="private">Private</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="position">Position</Label>
          <Input
            id="position"
            value={data.position}
            onChange={(e) => handleInputChange('position', e.target.value)}
            placeholder="Enter job position"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="employmentDuration">Employment Duration</Label>
          <Select onValueChange={(value) => handleInputChange('employmentDuration', value)}>
            <SelectTrigger>
              <SelectValue placeholder="Select duration" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="less-than-1">Less than 1 year</SelectItem>
              <SelectItem value="1-2">1-2 years</SelectItem>
              <SelectItem value="3-5">3-5 years</SelectItem>
              <SelectItem value="6-10">6-10 years</SelectItem>
              <SelectItem value="more-than-10">More than 10 years</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="salary">Salary</Label>
          <Input
            id="salary"
            value={data.salary}
            onChange={(e) => handleInputChange('salary', e.target.value)}
            placeholder="Enter monthly salary"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="typeOfSalary">Salary Frequency</Label>
          <Select onValueChange={(value) => handleInputChange('typeOfSalary', value)}>
            <SelectTrigger>
              <SelectValue placeholder="Select salary frequency" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="monthly">Monthly</SelectItem>
              <SelectItem value="bimonthly">Bimonthly</SelectItem>
              <SelectItem value="biweekly">Biweekly</SelectItem>
              <SelectItem value="weekly">Weekly</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
    </div>
  );
}