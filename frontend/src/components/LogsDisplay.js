import React from 'react';
import Table from 'react-bootstrap/Table';

function LogsDisplay({ logs }) {
  return (
    <div className="mt-4">
      <h2>Firewall Logs</h2>
      <Table striped bordered hover>
        <thead>
          <tr>
            <th>ID</th>
            <th>Payload</th>
            <th>Attack Type</th>
            <th>Time Detected</th>
            <th>Layer</th>
          </tr>
        </thead>
        <tbody>
          {logs.map((log) => (
            <tr key={log.id}>
              <td>{log.id}</td>
              <td>{log.payload}</td>
              <td>{log.attack_type}</td>
              <td>{log.time_detected}</td>
              <td>{log.layer}</td>
            </tr>
          ))}
        </tbody>
      </Table>
    </div>
  );
}

export default LogsDisplay;