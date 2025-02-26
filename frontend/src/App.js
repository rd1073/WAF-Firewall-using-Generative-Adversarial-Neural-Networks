import React, { useState, useEffect } from 'react';
import axios from 'axios';
import FirewallToggle from '../src/components/FirewallToggle';
import LogsDisplay from '../src/components/LogsDisplay';
import DetectionForm from '../src/components/DetectionForm';
import Alert from 'react-bootstrap/Alert';
import 'bootstrap/dist/css/bootstrap.min.css';
import { io } from 'socket.io-client';

function App() {
  const [logs, setLogs] = useState([]);
  const [firewallStatus, setFirewallStatus] = useState(true);
  const [notification, setNotification] = useState(null);



  useEffect(() => {
    const fetchFirewallStatus = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:5000/firewall-status');
        setFirewallStatus(response.data.status);
      } catch (error) {
        console.error('Error fetching firewall status:', error);
      }
    };

    fetchFirewallStatus();
  }, []);



  const fetchLogs = async () => {
    try {
      const response = await axios.get('http://127.0.0.1:5000/logs');
      setLogs(response.data);
      console.log(response.data); // Log the response for debugging

    } catch (error) {
      console.error('Error fetching logs:', error);
    }
  };

  useEffect(() => {
    fetchLogs();

   // Connect to the WebSocket server
   const socket = io('http://127.0.0.1:5000');

   // Listen for new detection events
   socket.on('new_detection', (data) => {
     setNotification(data); // Display the notification
     fetchLogs(); // Refresh the logs
   });

   // Clean up the WebSocket connection
   return () => {
     socket.disconnect();
   };
 }, []);

  const toggleFirewall = async (enabled) => {
    try {
      await axios.post('http://127.0.0.1:5000/toggle-firewall', { enabled });
      setFirewallStatus(enabled);
    } catch (error) {
      console.error('Error toggling firewall:', error);
    }
  };

  return (
    <div className="container mt-5">
      <h1 className="text-center mb-4">Firewall Dashboard</h1>
      {notification && (
        <Alert
          variant={notification.attack_type === 'Normal Request' ? 'success' : 'danger'}
          className="mb-4"
          onClose={() => setNotification(null)}
          dismissible
        >
          <h4>New Detection!</h4>
          <p><strong>Payload:</strong> {notification.payload}</p>
          <p><strong>Attack Type:</strong> {notification.attack_type}</p>
          <p><strong>Time Detected:</strong> {notification.time_detected}</p>
          <p><strong>Layer:</strong> {notification.layer}</p>
        </Alert>
      )}
       
      <div className="container mt-5">
      <h1 className="text-center mb-4">Firewall Dashboard</h1>
      <FirewallToggle firewallStatus={firewallStatus} toggleFirewall={toggleFirewall} />
    </div>
      <DetectionForm firewallStatus={firewallStatus} />
      <LogsDisplay logs={logs} />
    </div>
  );
}

export default App;