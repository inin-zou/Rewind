import type { GenerateResponse, SoundResponse } from "@/lib/api";

let pendingGeneration: Promise<GenerateResponse> | null = null;
let resolvedResult: GenerateResponse | null = null;

let pendingSoundGeneration: Promise<SoundResponse> | null = null;
let resolvedSoundResult: SoundResponse | null = null;

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

export function setPendingSoundGeneration(promise: Promise<SoundResponse>) {
  pendingSoundGeneration = promise;
  resolvedSoundResult = null;
  promise
    .then((result) => {
      resolvedSoundResult = result;
    })
    .catch(() => {
      // Error handled by the consumer
    });
}

export function getPendingSoundGeneration(): Promise<SoundResponse> | null {
  return pendingSoundGeneration;
}

export function getResolvedSoundResult(): SoundResponse | null {
  return resolvedSoundResult;
}

export function clearGeneration() {
  pendingGeneration = null;
  resolvedResult = null;
  pendingSoundGeneration = null;
  resolvedSoundResult = null;
}
