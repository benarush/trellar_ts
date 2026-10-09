/** Minimal logger (Python: ``logging.getLogger(__name__)``). Warnings go to ``console.warn``. */
export const logger = {
  warning(message: string, error?: unknown): void {
    if (error === undefined) {
      console.warn(`[trellar] ${message}`);
    } else {
      console.warn(`[trellar] ${message}`, error);
    }
  },
};
