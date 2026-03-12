// a snippet from the actual firm_tasks.c file, showing just the data filter task.
// Updated for ESKF (Error-State Extended Kalman Filter).

void filter_data_task(void *argument) {

  ESKF eskf;
  memset(&eskf, 0, sizeof(ESKF));
  ESKFRawData raw_data_instance;
  TaskCommandOption cmd_status = TASKCMD_SETUP;
  double last_time = 0.0;
  TickType_t start_time = xTaskGetTickCount();
  const TickType_t kalman_frequency = MAX_WAIT_TIME(firmSettings.frequency_hz);

  for (;;) {
    xQueueReceive(data_filter_command_queue, &cmd_status, 0);
    if (cmd_status == TASKCMD_SETUP || cmd_status == TASKCMD_MOCK_SETUP) {
      // making sure all sensors have delivered valid data before accumulating
      if (data_packet.data.data_packet.magnetic_field_x_microteslas == 0 ||
          data_packet.data.data_packet.pressure_pascals == 0) {
        vTaskDelay(1);
        continue;
      }

      // loop collecting data until enough is collected to get starting rotation and initial
      // altitude
      TickType_t accum_start = xTaskGetTickCount();
      while ((xTaskGetTickCount() - accum_start) <
             pdMS_TO_TICKS(KALMAN_FILTER_STARTUP_DELAY_TIME_MS)) {
        // orientation initialization relies on acceleration and magnetic field, initial altitude
        // uses raw pressure

        eskf_accumulate(&eskf, data_packet.data.data_packet.pressure_pascals,
                        &data_packet.data.data_packet.raw_acceleration_x_gs,
                        &data_packet.data.data_packet.magnetic_field_x_microteslas);

        // yield so sensor tasks can update data_packet with fresh readings
        vTaskDelay(pdMS_TO_TICKS(5));
      }

      // filter can initialize, and FIRM can go into live mode
      eskf_init(&eskf);
      if (cmd_status == TASKCMD_SETUP) {
        xQueueSend(system_request_queue, &(SystemRequest){SYSREQ_FINISH_SETUP}, portMAX_DELAY);
      }
      if (cmd_status == TASKCMD_MOCK_SETUP) {
        xQueueSend(system_request_queue, &(SystemRequest){SYSREQ_FINISH_MOCK_SETUP}, portMAX_DELAY);
      }

      // reset start_time so vTaskDelayUntil runs at the correct cadence
      start_time = xTaskGetTickCount();

      // set the last time to calculate the delta timestamp, minus some initial offset so that the
      // first iteration of the filter doesn't have an extremely small dt.
      last_time = data_packet.data.data_packet.timestamp_seconds - 0.005;
    }

    if (cmd_status != TASKCMD_SETUP && cmd_status != TASKCMD_MOCK_SETUP) {
      // setting the current data packet to the instance of data to be used in this next filter
      // update
      raw_data_instance.timestamp_seconds = data_packet.data.data_packet.timestamp_seconds;
      memcpy(&raw_data_instance.pressure_pascals, &data_packet.data.data_packet.pressure_pascals,
             sizeof(raw_data_instance) - sizeof(raw_data_instance.timestamp_seconds));
      double dt = raw_data_instance.timestamp_seconds - last_time;
      
      if (dt > 1e-6) {
        last_time = raw_data_instance.timestamp_seconds;
        /* ---- Build raw IMU readings (sensor frame) ---- */
        float accel_raw[3] = {raw_data_instance.raw_acceleration_x_gs,
                              raw_data_instance.raw_acceleration_y_gs,
                              raw_data_instance.raw_acceleration_z_gs};
        float gyro_raw[3] = {raw_data_instance.raw_angular_rate_x_deg_per_s,
                             raw_data_instance.raw_angular_rate_y_deg_per_s,
                             raw_data_instance.raw_angular_rate_z_deg_per_s};

        /* ---- Build control input: IMU ---- */
        float u[ESKF_CONTROL_DIM] = {
            accel_raw[0], accel_raw[1], accel_raw[2], gyro_raw[0], gyro_raw[1], gyro_raw[2],
        };

        /* ---- Predict ---- */
        eskf_predict(&eskf, u, (float)dt);

        /* ---- Build measurement: pressure + mag ---- */
        float z_raw[ESKF_MEASUREMENT_DIM] = {
            raw_data_instance.pressure_pascals,
            raw_data_instance.magnetic_field_x_microteslas,
            raw_data_instance.magnetic_field_y_microteslas,
            raw_data_instance.magnetic_field_z_microteslas,
        };
        eskf_set_measurement(&eskf, z_raw);

        /* ---- Update ---- */
        eskf_update(&eskf, eskf.z);

        /* ---- Copy estimates back to data packet ---- */
        memcpy(&data_packet.data.data_packet.est_position_x_meters, eskf.x_nom,
               ESKF_NOMINAL_DIM * sizeof(float));
      }

      vTaskDelayUntil(&start_time, kalman_frequency);
    }
  }
}
