import { useSyncExternalStore } from "react";

/** A tiny shared store, so the 3D scene can react to a prediction without React re-rendering every frame. */
export function createStore<T extends object>(initial: T) {
  let state = initial;
  const listeners = new Set<() => void>();
  return {
    get: () => state,
    set(patch: Partial<T>) {
      state = { ...state, ...patch };
      listeners.forEach((l) => l());
    },
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

export interface SceneState {
  /** RY angles of the last prediction, one per qubit; null until the first prediction. */
  angles: number[] | null;
  names: string[] | null;
  demo: boolean;
}

export const sceneStore = createStore<SceneState>({ angles: null, names: null, demo: false });

export function useScene(): SceneState {
  return useSyncExternalStore(sceneStore.subscribe, sceneStore.get, sceneStore.get);
}

/**
 * Scroll position as a continuous section index: 0 = hero, 1 = signals, 2 = qubits,
 * 3 = screening tool, 4 = results. Written by the scroll listener, read inside useFrame.
 */
export const scrollState = { anchor: 0 };
export const scrollListeners = new Set<() => void>();

/**
 * Where the pinned circuit story is: 0 = all qubits at |0>, 1 = features encoded,
 * 2 = entangled, 3 = measured. Mutated by the scroll timeline, read inside useFrame.
 */
export const circuitState = { step: 3 };
