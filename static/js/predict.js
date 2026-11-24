/**
 * HomeValue - Property Prediction Engine Client Script
 * Handles real-time inference, multi-model evaluation, and Chart.js comparison
 */

let predictionChartInstance = null;

// Property Presets
const PRESETS = {
  starter: {
    living_area_sqft: 1300,
    total_interior_sqft: 1320,
    footprint_sqft: 980,
    garage_capacity_cars: 1,
    lot_size_sqft: 3200,
    room_count_above_ground: 3
  },
  midsize: {
    living_area_sqft: 2400,
    total_interior_sqft: 2420,
    footprint_sqft: 1800,
    garage_capacity_cars: 2,
    lot_size_sqft: 5500,
    room_count_above_ground: 5
  },
  luxury: {
    living_area_sqft: 3600,
    total_interior_sqft: 3650,
    footprint_sqft: 2700,
    garage_capacity_cars: 3,
    lot_size_sqft: 8500,
    room_count_above_ground: 7
  }
};

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('prediction-form');
  const btnSubmit = document.getElementById('btn-submit');
  const btnReset = document.getElementById('btn-reset');
  const errorAlert = document.getElementById('form-error-alert');
  const modelRadioCards = document.querySelectorAll('.model-radio-card');
  const presetButtons = document.querySelectorAll('.preset-btn');

  // Handle Model Radio Selection UI
  modelRadioCards.forEach(card => {
    card.addEventListener('click', () => {
      modelRadioCards.forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      const radio = card.querySelector('input[type="radio"]');
      if (radio) {
        radio.checked = true;
      }
      // Re-trigger calculation if results are already shown
      triggerPrediction();
    });
  });

  // Handle Presets
  presetButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const pKey = btn.getAttribute('data-preset');
      const p = PRESETS[pKey];
      if (p) {
        for (const [key, val] of Object.entries(p)) {
          const input = document.getElementById(key);
          if (input) input.value = val;
        }
        triggerPrediction();
      }
    });
  });

  // Handle Form Submission
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    triggerPrediction();
  });

  // Handle Reset
  btnReset.addEventListener('click', () => {
    form.reset();
    errorAlert.style.display = 'none';
    const defaultRadio = document.querySelector('input[value="Ridge Regression"]');
    if (defaultRadio) {
      defaultRadio.checked = true;
      modelRadioCards.forEach(c => c.classList.remove('active'));
      defaultRadio.closest('.model-radio-card').classList.add('active');
    }
    triggerPrediction();
  });

  // Initial Calculation on Page Load
  triggerPrediction();

  function triggerPrediction() {
    errorAlert.style.display = 'none';
    errorAlert.innerHTML = '';
    btnSubmit.disabled = true;
    btnSubmit.textContent = 'Calculating Valuation...';

    // Collect Form Data
    const selectedModel = document.querySelector('input[name="model"]:checked')?.value || 'Ridge Regression';
    const payload = {
      model: selectedModel,
      living_area_sqft: parseFloat(document.getElementById('living_area_sqft').value),
      total_interior_sqft: parseFloat(document.getElementById('total_interior_sqft').value),
      footprint_sqft: parseFloat(document.getElementById('footprint_sqft').value),
      garage_capacity_cars: parseInt(document.getElementById('garage_capacity_cars').value, 10),
      lot_size_sqft: parseFloat(document.getElementById('lot_size_sqft').value),
      room_count_above_ground: parseInt(document.getElementById('room_count_above_ground').value, 10)
    };

    fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
      btnSubmit.disabled = false;
      btnSubmit.textContent = 'Calculate Property Valuation';

      if (data.status === 'success') {
        renderPredictionResults(data, payload);
      } else {
        showError(data.message, data.errors);
      }
    })
    .catch(err => {
      btnSubmit.disabled = false;
      btnSubmit.textContent = 'Calculate Property Valuation';
      showError('Unable to connect to prediction server. Check network or server status.');
    });
  }

  function renderPredictionResults(data, inputPayload) {
    // 1. Primary Price
    document.getElementById('result-primary-price').textContent = formatCurrency(data.valuation);
    document.getElementById('result-active-badge').textContent = data.selected_model;

    // 2. Spread / Consensus
    const spreadElem = document.getElementById('result-spread');
    spreadElem.textContent = `± ${formatCurrency(data.model_spread)} (${data.model_spread < 5000 ? 'High Model Consensus' : 'Moderate Variance'})`;

    // 3. Multi-model Tiles
    document.getElementById('val-linear').textContent = formatCurrency(data.comparisons['Linear Regression']);
    document.getElementById('val-ridge').textContent = formatCurrency(data.comparisons['Ridge Regression']);
    document.getElementById('val-lasso').textContent = formatCurrency(data.comparisons['Lasso Regression']);

    // 4. Update Summary Table
    const tbody = document.getElementById('summary-table-body');
    if (tbody) {
      tbody.innerHTML = `
        <tr><td>Living Area</td><td class="text-right"><strong>${formatCurrency(inputPayload.living_area_sqft)} sq. ft.</strong></td></tr>
        <tr><td>Total Interior</td><td class="text-right"><strong>${formatCurrency(inputPayload.total_interior_sqft)} sq. ft.</strong></td></tr>
        <tr><td>Footprint Area</td><td class="text-right"><strong>${formatCurrency(inputPayload.footprint_sqft)} sq. ft.</strong></td></tr>
        <tr><td>Garage Capacity</td><td class="text-right"><strong>${inputPayload.garage_capacity_cars} Cars</strong></td></tr>
        <tr><td>Lot Size</td><td class="text-right"><strong>${formatCurrency(inputPayload.lot_size_sqft)} sq. ft.</strong></td></tr>
        <tr><td>Rooms (Above Ground)</td><td class="text-right"><strong>${inputPayload.room_count_above_ground} Rooms</strong></td></tr>
      `;
    }

    // 5. Render / Update Chart.js Comparison Chart
    renderPredictionChart(data.comparisons, data.selected_model);
  }

  function renderPredictionChart(comparisons, activeModel) {
    const ctx = document.getElementById('predictionChart');
    if (!ctx) return;

    const labels = ['Linear (OLS)', 'Ridge (L2)', 'Lasso (L1)'];
    const values = [
      comparisons['Linear Regression'],
      comparisons['Ridge Regression'],
      comparisons['Lasso Regression']
    ];

    const backgroundColors = labels.map(label => {
      if ((label.includes('Ridge') && activeModel.includes('Ridge')) ||
          (label.includes('Lasso') && activeModel.includes('Lasso')) ||
          (label.includes('Linear') && activeModel.includes('Linear'))) {
        return '#26877D'; // Accent Teal
      }
      return '#233D5B'; // Secondary Navy
    });

    if (predictionChartInstance) {
      predictionChartInstance.data.datasets[0].data = values;
      predictionChartInstance.data.datasets[0].backgroundColor = backgroundColors;
      predictionChartInstance.update();
      return;
    }

    predictionChartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Valuation',
          data: values,
          backgroundColor: backgroundColors,
          borderRadius: 4,
          barPercentage: 0.55
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (context) => `Valuation: ${formatCurrency(context.raw)}`
            }
          }
        },
        scales: {
          y: {
            grid: { color: '#E4E8EC' },
            ticks: {
              font: { family: 'Inter', size: 11 },
              callback: (val) => formatCurrency(val)
            }
          },
          x: {
            grid: { display: false },
            ticks: { font: { family: 'Inter', size: 12, weight: 600 } }
          }
        }
      }
    });
  }

  function showError(msg, errors = []) {
    errorAlert.style.display = 'block';
    let html = `<strong>${msg}</strong>`;
    if (errors && errors.length > 0) {
      html += `<ul style="margin-top: 6px; margin-left: 18px;">${errors.map(e => `<li>${e}</li>`).join('')}</ul>`;
    }
    errorAlert.innerHTML = html;
  }
});
