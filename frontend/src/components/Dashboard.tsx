import React from 'react';
import { PredictTab } from './PredictTab';
import { MetricsTab } from './MetricsTab';
import { ModelsTab } from './ModelsTab';

interface DashboardProps {
  activeTab: string;
}

export const Dashboard: React.FC<DashboardProps> = ({ activeTab }) => {
  return (
    <div className="container" style={{ paddingBottom: '4rem' }}>
      {activeTab === 'predict' && <PredictTab />}
      {activeTab === 'metrics' && <MetricsTab />}
      {activeTab === 'models' && <ModelsTab />}
    </div>
  );
};
