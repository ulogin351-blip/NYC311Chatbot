import React from 'react';

const ChartComponent = ({ visualizationData }) => {
  if (!visualizationData) {
    return null;
  }

  // Debug: Log the received visualization data
  console.log('ChartComponent received data:', visualizationData);

  // Handle image-based visualizations
  if (visualizationData.type === 'image' && visualizationData.image_data) {
    return (
      <div className="visualization-container" style={{ padding: '20px' }}>
        <div className="chart-header">
          <h3 style={{ color: '#646cff', marginBottom: '10px' }}>
            {visualizationData.title || 'Data Visualization'}
          </h3>
          {visualizationData.description && (
            <p style={{ color: '#888', fontSize: '14px', marginBottom: '15px' }}>
              {visualizationData.description}
            </p>
          )}
        </div>
        <div className="image-chart" style={{ textAlign: 'center', marginBottom: '15px' }}>
          <img 
            src={visualizationData.image_data} 
            alt={visualizationData.title || 'Data Visualization'}
            style={{ 
              maxWidth: '100%', 
              height: 'auto',
              border: '1px solid #333',
              borderRadius: '8px',
              backgroundColor: 'white'
            }} 
          />
        </div>
        {/* Show additional insights if available */}
        {visualizationData.insights && (
          <div style={{ 
            background: 'rgba(100, 108, 255, 0.1)', 
            padding: '15px', 
            borderRadius: '8px',
            marginTop: '15px'
          }}>
            <h4 style={{ color: '#646cff', marginBottom: '10px' }}>Key Insights:</h4>
            <p style={{ color: '#ccc', fontSize: '14px', lineHeight: '1.5' }}>
              {visualizationData.insights}
            </p>
          </div>
        )}
      </div>
    );
  }

  // Handle text-based analysis
  if (visualizationData.type === 'text_analysis') {
    return (
      <div className="visualization-container" style={{ padding: '20px' }}>
        <div className="text-analysis" style={{ 
          background: 'rgba(100, 108, 255, 0.1)', 
          padding: '20px', 
          borderRadius: '8px',
          border: '1px solid #646cff'
        }}>
          <h3 style={{ color: '#646cff', marginBottom: '15px' }}>
            {visualizationData.title || 'Data Analysis'}
          </h3>
          <div style={{ color: '#ccc', lineHeight: '1.6' }}>
            {visualizationData.description}
          </div>
        </div>
      </div>
    );
  }

  // If no visualization data matches our supported formats, show a message
  return (
    <div className="visualization-container" style={{ padding: '20px' }}>
      <div style={{ 
        background: 'rgba(255, 99, 132, 0.1)', 
        padding: '20px', 
        borderRadius: '8px',
        border: '1px solid #ff6384'
      }}>
        <h3 style={{ color: '#ff6384', marginBottom: '10px' }}>
          Unsupported Visualization Format
        </h3>
        <p style={{ color: '#ccc', fontSize: '14px' }}>
          This visualization format is not supported. Please request an image-based visualization.
        </p>
      </div>
    </div>
  );
};

export default ChartComponent;
