document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const fiberLength = document.getElementById('fiberLength').value;
    const data = { fiber_length: parseFloat(fiberLength) };

    // Example: Send to backend API
    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    const result = await response.json();

    // Display results
    document.getElementById('results').innerHTML = `
        <p class="mt-2">Predicted Key Rate: ${result.key_rate}</p>
        <p>Optimal Parameter: ${result.parameter}</p>
    `;
});