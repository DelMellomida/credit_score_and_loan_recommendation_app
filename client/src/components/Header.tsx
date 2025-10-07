"use client";

import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { User, LogOut } from 'lucide-react';
import { Button } from './ui/button';
import { Popover, PopoverContent, PopoverTrigger } from './ui/popover';


export function Header() {
  const [isOpen, setIsOpen] = useState(false);
  const { logout, user } = useAuth();

  const currentDateTime = new Date().toLocaleString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: true
  });

  const handleLogout = () => {
    logout();
    setIsOpen(false);
  };

  return (
    <header className="bg-red-600 text-white px-6 py-4 shadow-lg">
      <div className="flex justify-between items-center max-w-9xl mx-auto">
        <div className="flex items-center">
          <h1 className="text-2xl font-medium">Best Loan</h1>
        </div>
        
        <div className="flex items-center">
          <Popover open={isOpen} onOpenChange={setIsOpen}>
            <PopoverTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                className="text-white hover:bg-red-700 p-2"
              >
                <User className="h-6 w-6" />
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-64 mr-6" align="end">
              <div className="space-y-3">
                <div className="text-sm text-gray-600">
                  <p className="font-medium">Current Time</p>
                  <p>{currentDateTime}</p>
                </div>
                <hr />
                <Button
                  onClick={handleLogout}
                  variant="outline"
                  className="w-full flex items-center gap-2 text-red-600 border-red-600 hover:bg-red-50"
                  disabled={!user}
                >
                  <LogOut className="h-4 w-4" />
                  Logout
                </Button>
              </div>
            </PopoverContent>
          </Popover>
        </div>
      </div>
    </header>
  );
}