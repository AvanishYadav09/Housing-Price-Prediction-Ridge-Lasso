/**
 * HomeValue - Main JavaScript
 * Shared utilities and global helpers
 */

// Format numbers as currency style (without forcing arbitrary symbol)
function formatCurrency(num) {
  if (isNaN(num)) return "0";
  return Math.round(num).toLocaleString('en-US');
}

// Format floating numbers with fixed precision
function formatNumber(num, decimals = 2) {
  if (isNaN(num)) return "0";
  return Number(num).toFixed(decimals);
}

document.addEventListener('DOMContentLoaded', () => {
  // Console identity
  console.log("HomeValue Housing Price Prediction System initialized.");
});
