export const ENV_TRELLAR_API_KEY = "TRELLAR_API_KEY";

// export const DEFAULT_ENDPOINT = "https://api.trellar.io/";
export const DEFAULT_ENDPOINT = "http://localhost:8001";

export function getEnvApiKey(): string | undefined {
  const value = process.env[ENV_TRELLAR_API_KEY];
  return value === undefined || value === "" ? undefined : value;
}
