import React, { useState } from 'react';
import axios from 'axios';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Alert from 'react-bootstrap/Alert';

function DetectionForm({ firewallStatus }) {
  const [sentence, setSentence] = useState('');
  const [detectionResult, setDetectionResult] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://127.0.0.1:5000/detect', { sentence });
      setDetectionResult(response.data.detection);
    } catch (error) {
      console.error('Error detecting attack:', error);
    }
  };

  return (
    <div className="mt-4">
      {/* <h2>Firewall Detection</h2>
      <Form onSubmit={handleSubmit}>
        <Form.Group controlId="sentence">
          <Form.Label>Enter a sentence to detect:</Form.Label>
          <Form.Control
            type="text"
            value={sentence}
            onChange={(e) => setSentence(e.target.value)}
            placeholder="Enter a sentence"
          />
        </Form.Group>
        <Button variant="primary" type="submit" disabled={!firewallStatus}>
          Detect
        </Button>
      </Form> */}
      {detectionResult && (
        <Alert variant={detectionResult === 'Normal Request' ? 'success' : 'danger'} className="mt-3">
          Detection Result: {detectionResult}
        </Alert>
      )}
    </div>
  );
}

export default DetectionForm;