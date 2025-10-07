// API utility functions for Next.js client
// Uses NEXT_PUBLIC_API_URL from .env.local

const API_URL = process.env.NEXT_PUBLIC_API_URL;

// Helper for requests
async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_URL}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    },
    credentials: 'include', // for cookies if needed
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ message: res.statusText }));
    throw new Error(error.message || 'API error');
  }
  return res.json();
}

// Auth endpoints
export async function login(username: string, password: string) {
  // FastAPI expects x-www-form-urlencoded for OAuth2
  const body = new URLSearchParams({ username, password });
  const res = await fetch(`${API_URL}/auth/login`, {
    method: 'POST',
    body,
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    credentials: 'include',
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ message: res.statusText }));
    throw new Error(error.message || 'Login failed');
  }
  return res.json();
}

export async function signup(email: string, full_name: string, password: string) {
  return request('/auth/signup', {
    method: 'POST',
    body: JSON.stringify({ email, full_name, password }),
  });
}

// Loan application endpoints
export async function createLoanApplication(data: any, token?: string) {
  return request('/loans/applications', {
    method: 'POST',
    body: JSON.stringify(data),
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
}

export async function getMyApplications(token?: string) {
  return request('/loans/my-applications', {
    method: 'GET',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
}

export async function getLoanApplication(applicationId: string, token?: string) {
  return request(`/loans/applications/${applicationId}`, {
    method: 'GET',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
}

// Demo loan application (for presentations/testing)
export async function createDemoLoanApplication(params: Record<string, any>, token?: string) {
  // params should match backend demo endpoint fields
  const searchParams = new URLSearchParams(params).toString();
  return request(`/loans/applications/demo?${searchParams}`, {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
}

// Update application status
export async function updateApplicationStatus(applicationId: string, statusUpdate: Record<string, string>, token?: string) {
  return request(`/loans/applications/${applicationId}/status`, {
    method: 'PUT',
    body: JSON.stringify(statusUpdate),
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
}

// Regenerate loan recommendations
export async function regenerateLoanRecommendations(applicationId: string, token?: string) {
  return request(`/loans/applications/${applicationId}/regenerate-recommendations`, {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
}

// Delete loan application
export async function deleteLoanApplication(applicationId: string, token?: string) {
  return request(`/loans/applications/${applicationId}`, {
    method: 'DELETE',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
}

// Health check endpoints
export async function healthCheck() {
  return request('/loans/health', { method: 'GET' });
}

export async function getServiceStatus(token?: string) {
  return request('/loans/service-status', {
    method: 'GET',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
}
