import React from 'react';
import Button from 'react-bootstrap/Button';

function FirewallToggle({ firewallStatus, toggleFirewall }) {
  return (
    <div className="mb-4">
      <h2>Firewall Status: {firewallStatus ? 'ON' : 'OFF'}</h2>
      <Button
        variant={firewallStatus ? 'danger' : 'success'}
        onClick={() => toggleFirewall(!firewallStatus)}
      >
        Turn {firewallStatus ? 'OFF' : 'ON'}
      </Button>
    </div>
  );
}

export default FirewallToggle;