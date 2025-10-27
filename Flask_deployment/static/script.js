// Prediction polling configuration
const POLL_INTERVAL = 200; // milliseconds (5 FPS for UI updates)
let lastUpdateTime = Date.now();
let frameCount = 0;

// Start polling for predictions
async function pollPredictions() {
    try {
        const response = await fetch('/predict');
        const data = await response.json();
        
        // Update FPS counter
        frameCount++;
        const now = Date.now();
        if (now - lastUpdateTime >= 1000) {
            const fps = Math.round(frameCount / ((now - lastUpdateTime) / 1000));
            document.getElementById('fps').textContent = fps;
            frameCount = 0;
            lastUpdateTime = now;
        }
        
        // Update UI with prediction
        updatePredictionDisplay(data);
        
    } catch (error) {
        console.error('Error fetching prediction:', error);
        updateStatus('Error connecting to server', 'inactive');
    }
    
    // Continue polling
    setTimeout(pollPredictions, POLL_INTERVAL);
}

// Update prediction display
function updatePredictionDisplay(data) {
    const letterElement = document.getElementById('predictedLetter');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceValue = document.getElementById('confidenceValue');
    
    if (data.label && data.confidence > 0) {
        // Update letter
        letterElement.textContent = data.label;
        letterElement.style.color = getConfidenceColor(data.confidence);
        
        // Update confidence bar
        const confidencePercent = (data.confidence * 100).toFixed(1);
        confidenceFill.style.width = confidencePercent + '%';
        confidenceValue.textContent = confidencePercent + '%';
        
        // Update status
        updateStatus('Hand detected - Predicting...', 'active');
    } else {
        // No hand detected
        letterElement.textContent = '-';
        letterElement.style.color = '#ccc';
        confidenceFill.style.width = '0%';
        confidenceValue.textContent = '0%';
        updateStatus('Waiting for hand...', 'inactive');
    }
}

// Get color based on confidence level
function getConfidenceColor(confidence) {
    if (confidence >= 0.8) return '#4caf50'; // Green
    if (confidence >= 0.6) return '#ff9800'; // Orange
    return '#f44336'; // Red
}

// Update status indicator
function updateStatus(text, state) {
    const statusText = document.getElementById('statusText');
    const indicator = document.querySelector('.status-indicator');
    
    statusText.textContent = text;
    indicator.className = 'status-indicator ' + state;
}

// Start the application
document.addEventListener('DOMContentLoaded', () => {
    console.log('ASL Recognition started');
    pollPredictions();
});
