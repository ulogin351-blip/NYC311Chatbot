import React from 'react';

const DataTable = ({ visualizationData }) => {
  if (!visualizationData || visualizationData.type !== 'table') {
    return null;
  }

  const { data, title } = visualizationData;
  const { columns, rows } = data;

  return (
    <div className="data-table-container">
      <h3 className="table-title">{title}</h3>
      <div className="table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              {columns.map((column, index) => (
                <th key={index}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.map((cell, cellIndex) => (
                  <td key={cellIndex}>
                    {typeof cell === 'number' ? cell.toLocaleString() : cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default DataTable;