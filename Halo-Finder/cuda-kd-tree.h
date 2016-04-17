void CopyDataToDevice(kdNode *dataArray, int *resultArray, int nParticles);
void ComputeResultArray(float linkingLength, int nParticles, int kdRoot);
void FetchDeviceResultArray(int *resultArray, int nParticles);
void RefreshDeviceResultArray(int *resultArray, int nParticles);
void ReleaseDeviceMemory();
double TimeDiff(timeval t1, timeval t2);