import React from 'react';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Textarea } from '../ui/textarea';
import { OtherData } from '../../app/page';

interface OtherDataFormProps {
  data: OtherData;
  updateData: (data: Partial<OtherData>) => void;
}

export function OtherDataForm({ data, updateData }: OtherDataFormProps) {
  const handleInputChange = (field: keyof OtherData, value: string) => {
    updateData({ [field]: value });
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="communityPosition">Community Role</Label>
          <Select onValueChange={(value) => handleInputChange('communityPosition', value)}>
            <SelectTrigger>
              <SelectValue placeholder="Select community role" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">None</SelectItem>
              <SelectItem value="member">Member</SelectItem>
              <SelectItem value="leader">Leader</SelectItem>
              <SelectItem value="multiple_leader">Multiple Leader</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="paluwagaParticipation">Paluwagan Participation</Label>
          <Select onValueChange={(value) => handleInputChange('paluwagaParticipation', value)}>
            <SelectTrigger>
              <SelectValue placeholder="Select participation" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="never">Never</SelectItem>
              <SelectItem value="rarely">Rarely</SelectItem>
              <SelectItem value="sometimes">Sometimes</SelectItem>
              <SelectItem value="frequently">Frequently</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="space-y-2">
        <Label htmlFor="otherIncomeSources">Other Income Source</Label>
        <Select onValueChange={(value) => handleInputChange('otherIncomeSources', value)}>
          <SelectTrigger>
            <SelectValue placeholder="Select other income source" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="none">None</SelectItem>
            <SelectItem value="ofw_remittance">OFW Remittance</SelectItem>
            <SelectItem value="freelance">Freelance</SelectItem>
            <SelectItem value="business">Business</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <Label htmlFor="disasterPreparednessStrategy">Disaster Preparedness</Label>
        <Select onValueChange={(value) => handleInputChange('disasterPreparednessStrategy', value)}>
          <SelectTrigger>
            <SelectValue placeholder="Select disaster preparedness" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="none">None</SelectItem>
            <SelectItem value="savings">Savings</SelectItem>
            <SelectItem value="insurance">Insurance</SelectItem>
            <SelectItem value="community_plan">Community Plan</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}