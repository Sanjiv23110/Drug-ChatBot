/**
 * Type-safe API client for Solomind Backend
 * Auto-generated types from OpenAPI spec
 */

import type { components, operations } from './api-types';

// Type shortcuts for convenience
export type QueryRequest = components['schemas']['QueryRequest'];
export type QueryResponse = components['schemas']['QueryResponse'];

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Type-safe chat query
 * Usage:
 *   const response = await chatQuery({ query: "What is Lisinopril?" });
 *   // TypeScript knows response.answer, response.status, etc.
 */
export async function chatQuery(
  request: Omit<QueryRequest, 'user_id' | 'session_id'> & Partial<Pick<QueryRequest, 'user_id' | 'session_id'>>
): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query: request.query,
      user_id: request.user_id || 'user',
      session_id: request.session_id || 'default',
    }),
  });

  if (!response.ok) {
    if (response.status === 422) {
      const error = await response.json();
      throw new Error(`Validation Error: ${JSON.stringify(error.detail)}`);
    }
    throw new Error(`HTTP Error ${response.status}: ${response.statusText}`);
  }

  const data: QueryResponse = await response.json();
  return data;
}
