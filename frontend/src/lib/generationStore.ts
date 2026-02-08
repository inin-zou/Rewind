import type { GenerateResponse } from "@/lib/api";

let pendingGeneration: Promise<GenerateResponse> | null = null;
let resolvedResult: GenerateResponse | null = null;

export function setPendingGeneration(promise: Promise<GenerateResponse>) {
  pendingGeneration = promise;
  resolvedResult = null;
  promise
    .then((result) => {
      resolvedResult = result;
    })
    .catch(() => {
      // Error handled by the consumer
    });
}

export function getPendingGeneration(): Promise<GenerateResponse> | null {
  return pendingGeneration;
}

export function getResolvedResult(): GenerateResponse | null {
  return resolvedResult;
}

export function clearGeneration() {
  pendingGeneration = null;
  resolvedResult = null;
}
