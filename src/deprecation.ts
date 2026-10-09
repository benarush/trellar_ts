const warned = new Set<string>();

/**
 * Build a deprecated alias of ``fn``: it emits a Node ``DeprecationWarning``
 * (once per old name) and forwards every call to the new function.
 */
export function deprecatedAlias<A extends unknown[], R>(
  oldName: string,
  newName: string,
  fn: (...args: A) => R,
): (...args: A) => R {
  return (...args: A): R => {
    if (!warned.has(oldName)) {
      warned.add(oldName);
      process.emitWarning(
        `${oldName}() is deprecated and will be removed in a future release; use ${newName}() instead.`,
        "DeprecationWarning",
      );
    }
    return fn(...args);
  };
}

/** Test helper: forget which deprecation warnings were already emitted. */
export function resetDeprecationWarnings(): void {
  warned.clear();
}
