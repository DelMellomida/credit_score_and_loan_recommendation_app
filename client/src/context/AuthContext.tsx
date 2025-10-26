"use client";
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { login as apiLogin, signup as apiSignup } from '../lib/api';

interface AuthUser {
  email: string;
  full_name: string;
  token: string;
}

interface AuthContextType {
  user: AuthUser | null;
  login: (username: string, password: string) => Promise<void>;
  signup: (email: string, full_name: string, password: string) => Promise<void>;
  logout: () => void;
  loading: boolean;
  error: string | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Restore user from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem('authUser');
    if (stored) {
      setUser(JSON.parse(stored));
    }
  }, []);

  // Save user to localStorage when changed
  useEffect(() => {
    if (user) {
      localStorage.setItem('authUser', JSON.stringify(user));
    } else {
      localStorage.removeItem('authUser');
    }
  }, [user]);

  const login = async (username: string, password: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiLogin(username, password);
      console.log('Auth response:', res);
      
      // Make sure we have a token
      if (!res.access_token) {
        throw new Error('No access token returned from server');
      }

      // Store just the raw token value
      const token = res.access_token;
      console.log('Setting auth token:', '<token>');
      
      setUser({
        email: username,
        full_name: res.full_name || username,
        token: token, // Store just the raw token value
      });
    } catch (err: any) {
      console.error('Login error:', err);
      setError(err.message || 'Login failed');
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  const signup = async (email: string, full_name: string, password: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiSignup(email, full_name, password);
      // Server returns user info, but not token; require login after signup
      setUser(null);
    } catch (err: any) {
      setError(err.message || 'Signup failed');
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    setUser(null);
    setError(null);
    localStorage.removeItem('authUser');
  };

  return (
    <AuthContext.Provider value={{ user, login, signup, logout, loading, error }}>
      {children}
    </AuthContext.Provider>
  );
};

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within AuthProvider');
  return context;
}
