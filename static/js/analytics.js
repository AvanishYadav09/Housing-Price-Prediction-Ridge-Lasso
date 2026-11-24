/**
 * HomeValue - Analytics Dashboard Visualizations
 * Fetches computed ML metrics and renders Chart.js charts
 */

document.addEventListener('DOMContentLoaded', () => {
  fetch('/api/analytics')
    .then(res => res.json())
    .then(payload => {
      if (payload.status === 'success') {
        const data = payload.data;
        renderModelPerformanceChart(data.test_evaluations);
        renderVifChart(data.vif_results);
        renderActualVsPredChart(data.scatter_sample);
        renderResidualChart(data.residual_distribution);
      }
    })
    .catch(err => {
      console.error('Failed to load analytics payload:', err);
    });
});

// 1. Model Performance (R2 and RMSE)
function renderModelPerformanceChart(evals) {
  const ctx = document.getElementById('modelPerfChart');
  if (!ctx) return;

  const models = ['Linear (OLS)', 'Ridge (L2)', 'Lasso (L1)'];
  const r2Scores = [
    evals['Linear Regression'].r2,
    evals['Ridge Regression'].r2,
    evals['Lasso Regression'].r2
  ];
  const rmseScores = [
    evals['Linear Regression'].rmse,
    evals['Ridge Regression'].rmse,
    evals['Lasso Regression'].rmse
  ];

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: models,
      datasets: [
        {
          label: 'Test R² Score',
          data: r2Scores,
          backgroundColor: '#26877D',
          borderRadius: 4,
          yAxisID: 'yR2',
          barPercentage: 0.5
        },
        {
          label: 'Test RMSE',
          data: rmseScores,
          backgroundColor: '#172B45',
          borderRadius: 4,
          yAxisID: 'yRMSE',
          barPercentage: 0.5
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { font: { family: 'Inter', size: 12 } }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              if (ctx.dataset.yAxisID === 'yR2') {
                return `R² Score: ${Number(ctx.raw).toFixed(4)}`;
              }
              return `RMSE: ${Math.round(ctx.raw).toLocaleString()}`;
            }
          }
        }
      },
      scales: {
        yR2: {
          type: 'linear',
          position: 'left',
          min: 0.95,
          max: 1.0,
          ticks: { font: { family: 'Inter', size: 11 } },
          grid: { color: '#E4E8EC' },
          title: { display: true, text: 'R² Score', font: { size: 11, weight: 600 } }
        },
        yRMSE: {
          type: 'linear',
          position: 'right',
          min: 20000,
          max: 26000,
          ticks: { font: { family: 'Inter', size: 11 } },
          grid: { display: false },
          title: { display: true, text: 'RMSE', font: { size: 11, weight: 600 } }
        },
        x: {
          grid: { display: false },
          ticks: { font: { family: 'Inter', size: 12, weight: 600 } }
        }
      }
    }
  });
}

// 2. Variance Inflation Factor (VIF) Bar Chart
function renderVifChart(vifData) {
  const ctx = document.getElementById('vifChart');
  if (!ctx) return;

  const labels = [
    'Living Area',
    'Total Interior',
    'Footprint',
    'Garage',
    'Lot Size',
    'Rooms'
  ];
  
  const rawKeys = [
    'living_area_sqft',
    'total_interior_sqft',
    'footprint_sqft',
    'garage_capacity_cars',
    'lot_size_sqft',
    'room_count_above_ground'
  ];

  const values = rawKeys.map(k => vifData[k] || 1.0);
  const colors = values.map(v => v > 10 ? '#C53929' : '#26877D');

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'VIF Score',
        data: values,
        backgroundColor: colors,
        borderRadius: 4,
        barPercentage: 0.55
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `VIF Score: ${Number(ctx.raw).toFixed(2)} ${ctx.raw > 10 ? '(Severe Collinearity)' : '(Normal)'}`
          }
        }
      },
      scales: {
        x: {
          type: 'logarithmic',
          grid: { color: '#E4E8EC' },
          ticks: {
            font: { family: 'Inter', size: 11 },
            callback: (val) => val
          },
          title: { display: true, text: 'VIF Score (Logarithmic Scale)', font: { size: 11, weight: 600 } }
        },
        y: {
          grid: { display: false },
          ticks: { font: { family: 'Inter', size: 11, weight: 600 } }
        }
      }
    }
  });
}

// 3. Actual vs. Predicted Price Scatter Chart
function renderActualVsPredChart(scatterSample) {
  const ctx = document.getElementById('actualVsPredChart');
  if (!ctx) return;

  const points = scatterSample.map(p => ({
    x: p.actual,
    y: p.pred_ridge
  }));

  const minVal = 200000;
  const maxVal = 900000;

  new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Ridge Predictions',
          data: points,
          backgroundColor: 'rgba(38, 135, 125, 0.65)',
          borderColor: '#26877D',
          pointRadius: 4,
          pointHoverRadius: 6
        },
        {
          label: 'Perfect Fit (Identity)',
          data: [{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }],
          type: 'line',
          borderColor: '#C53929',
          borderDash: [5, 5],
          fill: false,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { font: { family: 'Inter', size: 12 } }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `Actual: ${Math.round(ctx.parsed.x).toLocaleString()} | Pred: ${Math.round(ctx.parsed.y).toLocaleString()}`
          }
        }
      },
      scales: {
        x: {
          min: minVal,
          max: maxVal,
          grid: { color: '#E4E8EC' },
          ticks: {
            font: { family: 'Inter', size: 11 },
            callback: (v) => `${Math.round(v/1000)}k`
          },
          title: { display: true, text: 'Actual Sale Price', font: { size: 11, weight: 600 } }
        },
        y: {
          min: minVal,
          max: maxVal,
          grid: { color: '#E4E8EC' },
          ticks: {
            font: { family: 'Inter', size: 11 },
            callback: (v) => `${Math.round(v/1000)}k`
          },
          title: { display: true, text: 'Predicted Sale Price', font: { size: 11, weight: 600 } }
        }
      }
    }
  });
}

// 4. Residual Distribution Histogram
function renderResidualChart(residualHist) {
  const ctx = document.getElementById('residualChart');
  if (!ctx) return;

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: residualHist.bin_labels,
      datasets: [{
        label: 'Prediction Errors (Residuals)',
        data: residualHist.counts,
        backgroundColor: '#233D5B',
        borderRadius: 3,
        barPercentage: 0.95,
        categoryPercentage: 1.0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => `Error Interval: ${items[0].label}`,
            label: (ctx) => `Frequency: ${ctx.raw} observations`
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { font: { family: 'Inter', size: 10 }, maxRotation: 45 }
        },
        y: {
          grid: { color: '#E4E8EC' },
          ticks: { font: { family: 'Inter', size: 11 } },
          title: { display: true, text: 'Frequency (Count)', font: { size: 11, weight: 600 } }
        }
      }
    }
  });
}
