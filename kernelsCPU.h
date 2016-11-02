void innerProdCPU(float *aa, float *bb, float *cc)
{
   float temp = 0;
   for (int index = 0; index < N; index++)
   {
      temp += aa[index]*bb[index];
   }

   *cc = temp;
}
