/*
 * Standalone ESKF test harness.
 *
 * Reads a pre-processed CSV (produced by scripts/test_eskf_c.py) where each
 * row is one timestep with:
 *
 *   dt, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z,
 *   pressure, mag_x, mag_y, mag_z
 *
 * - accel is in g-units (sensor frame), with file-level calibration already
 *   applied.
 * - gyro is in deg/s (sensor frame), with file-level calibration already
 *   applied.
 * - mag is the raw value (NOT normalised) after hard/soft iron calibration.
 *
 * The first row is used for eskf_init (initial orientation + pressure).
 *
 * Writes test_output.csv with one row per step:
 *   step, dt, state, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
 *   qw, qx, qy, qz, mahal
 *
 * Compile:
 *   gcc -O2 -I ukf_data_processing \
 *       ukf_data_processing/matrixhelper.c \
 *       ukf_data_processing/kalman_filter_config.c \
 *       ukf_data_processing/state_machine.c \
 *       ukf_data_processing/ukf_functions.c \
 *       ukf_data_processing/unscented_kalman_filter.c \
 *       test_eskf.c -lm -o test_eskf
 */

#include "unscented_kalman_filter.h"
#include "ukf_functions.h"
#include "state_machine.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_ROWS 200000

int main(int argc, char **argv) {
  const char *input_file  = (argc > 1) ? argv[1] : "test_input.csv";
  const char *output_file = (argc > 2) ? argv[2] : "test_output.csv";

  FILE *fin = fopen(input_file, "r");
  if (!fin) {
    fprintf(stderr, "Cannot open %s\n", input_file);
    return 1;
  }

  /* Skip header line */
  char line[1024];
  if (!fgets(line, sizeof(line), fin)) {
    fclose(fin);
    return 1;
  }

  /* Read all rows into memory */
  typedef struct {
    float dt;
    float accel[3];
    float gyro[3];
    float pressure;
    float mag[3];
  } Row;

  Row *rows = (Row *)malloc(sizeof(Row) * MAX_ROWS);
  int n_rows = 0;

  while (fgets(line, sizeof(line), fin) && n_rows < MAX_ROWS) {
    Row *r = &rows[n_rows];
    int n = sscanf(line, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
                   &r->dt,
                   &r->accel[0], &r->accel[1], &r->accel[2],
                   &r->gyro[0],  &r->gyro[1],  &r->gyro[2],
                   &r->pressure,
                   &r->mag[0],   &r->mag[1],   &r->mag[2]);
    if (n == 11) n_rows++;
  }
  fclose(fin);

  if (n_rows < 2) {
    fprintf(stderr, "Need at least 2 data rows, got %d\n", n_rows);
    free(rows);
    return 1;
  }

  printf("Loaded %d rows from %s\n", n_rows, input_file);

  /* ---- Initialise ESKF ---- */
  ESKF eskf;
  Row *r0 = &rows[0];
  eskf_init(&eskf, r0->pressure, r0->accel, r0->mag);

  printf("Initial quaternion: [%.6f, %.6f, %.6f, %.6f]\n",
         eskf.x_nom[ESKF_QUAT_W], eskf.x_nom[ESKF_QUAT_X],
         eskf.x_nom[ESKF_QUAT_Y], eskf.x_nom[ESKF_QUAT_Z]);
  printf("mag_world: [%.6f, %.6f, %.6f]\n",
         eskf.mag_world[0], eskf.mag_world[1], eskf.mag_world[2]);

  /* ---- Open output ---- */
  FILE *fout = fopen(output_file, "w");
  if (!fout) {
    fprintf(stderr, "Cannot open %s for writing\n", output_file);
    free(rows);
    return 1;
  }
  fprintf(fout, "step,dt,state,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,"
                "qw,qx,qy,qz,mahal\n");

  float max_alt = 0.0f, max_vel = 0.0f;

  /* ---- Main loop (starting from row 0: first row is both init and first step) ---- */
  for (int i = 0; i < n_rows; i++) {
    Row *r = &rows[i];
    float dt = r->dt;
    if (dt < 1e-9f && i > 0) continue;
    /* use first row dt as-is (it will be the one the Python script computed) */

    /* During INIT: accumulate raw IMU for bias estimation */
    if (eskf.flight_state == ESKF_STATE_INIT) {
      eskf_accumulate(&eskf, r->accel, r->gyro);
    }

    /* Build control input: subtract biases */
    float u[ESKF_CONTROL_DIM] = {
      r->accel[0] - eskf.accel_bias[0],
      r->accel[1] - eskf.accel_bias[1],
      r->accel[2] - eskf.accel_bias[2],
      r->gyro[0]  - eskf.gyro_bias[0],
      r->gyro[1]  - eskf.gyro_bias[1],
      r->gyro[2]  - eskf.gyro_bias[2],
    };

    /* Predict */
    eskf_predict(&eskf, u, dt);

    /* Build measurement + normalise mag */
    float z_raw[4] = {r->pressure, r->mag[0], r->mag[1], r->mag[2]};
    eskf_set_measurement(&eskf, z_raw);

    /* Update */
    eskf_update(&eskf, eskf.z);

    /* State machine transition */
    eskf_state_update(&eskf);

    /* Track max */
    if (eskf.x_nom[ESKF_POS_Z] > max_alt)
      max_alt = eskf.x_nom[ESKF_POS_Z];
    float speed = sqrtf(eskf.x_nom[ESKF_VEL_X] * eskf.x_nom[ESKF_VEL_X] +
                        eskf.x_nom[ESKF_VEL_Y] * eskf.x_nom[ESKF_VEL_Y] +
                        eskf.x_nom[ESKF_VEL_Z] * eskf.x_nom[ESKF_VEL_Z]);
    if (speed > max_vel)
      max_vel = speed;

    /* Write output */
    fprintf(fout, "%d,%.9f,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
            i, dt, (int)eskf.flight_state,
            eskf.x_nom[ESKF_POS_X], eskf.x_nom[ESKF_POS_Y], eskf.x_nom[ESKF_POS_Z],
            eskf.x_nom[ESKF_VEL_X], eskf.x_nom[ESKF_VEL_Y], eskf.x_nom[ESKF_VEL_Z],
            eskf.x_nom[ESKF_QUAT_W], eskf.x_nom[ESKF_QUAT_X],
            eskf.x_nom[ESKF_QUAT_Y], eskf.x_nom[ESKF_QUAT_Z],
            eskf.mahalanobis_dist);
  }

  fclose(fout);
  free(rows);

  printf("\nResults written to %s\n", output_file);
  printf("Max altitude: %.2f m\n", max_alt);
  printf("Max speed:    %.2f m/s\n", max_vel);
  printf("Final state:  %d\n", (int)eskf.flight_state);
  printf("Accel bias:   [%.6f, %.6f, %.6f]\n",
         eskf.accel_bias[0], eskf.accel_bias[1], eskf.accel_bias[2]);
  printf("Gyro bias:    [%.6f, %.6f, %.6f]\n",
         eskf.gyro_bias[0], eskf.gyro_bias[1], eskf.gyro_bias[2]);
  return 0;
}
