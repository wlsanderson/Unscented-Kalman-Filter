// a snippet from the actual firm_tasks.c file, showing just the data filter task.
// Updated for ESKF (Error-State Extended Kalman Filter).

void filter_data_task(void *argument) {

  ESKF eskf;
  memset(&eskf, 0, sizeof(ESKF));
  ESKFRawData raw_data_instance;
  TaskCommandOption cmd_status = TASKCMD_SETUP;
  float last_time = 0.0F;

  for (;;) {
    xQueueReceive(data_filter_command_queue, &cmd_status, 0);
    if (cmd_status == TASKCMD_SETUP || cmd_status == TASKCMD_MOCK_SETUP) {
      // making sure data has been collected
      if (data_packet.data.data_packet.magnetic_field_x_microteslas == 0) {
        vTaskDelay(pdMS_TO_TICKS(10));
        continue;
      }
      if (data_packet.data.data_packet.pressure_pascals == 0) {
        vTaskDelay(pdMS_TO_TICKS(10));
        continue;
      }

      // Reset start_time right before we begin accumulating data for 2 seconds
      TickType_t start_time = xTaskGetTickCount();

      // loop collecting data until enough is collected to get starting rotation and initial
      // altitude. 100ms delay allows sensors to collect intitial data
      vTaskDelay(pdMS_TO_TICKS(100));
      while ((xTaskGetTickCount() - start_time) <
             pdMS_TO_TICKS(KALMAN_FILTER_STARTUP_DELAY_TIME_MS)) {
        vTaskDelay(pdMS_TO_TICKS(5));
        // orientation initialization relies on acceleration and magnetic field, initial altitude
        // uses raw pressure
        DataPacket *p = &data_packet.data.data_packet;
        eskf_accumulate(p->pressure_pascals, &p->raw_acceleration_x_gs,
                        &p->magnetic_field_x_microteslas);
      }

      // filter can initialize, and FIRM can go into live mode
      if (cmd_status == TASKCMD_SETUP) {
        xQueueSend(system_request_queue, &(SystemRequest){SYSREQ_FINISH_SETUP}, portMAX_DELAY);
        cmd_status = TASKCMD_LIVE;
      }
      if (cmd_status == TASKCMD_MOCK_SETUP) {
        xQueueSend(system_request_queue, &(SystemRequest){SYSREQ_FINISH_MOCK_SETUP}, portMAX_DELAY);
        cmd_status = TASKCMD_MOCK;
      }
      eskf_init(&eskf);
      // set the last time to calculate the delta timestamp, minus some initial offset so that the
      // first iteration of the filter doesn't have an extremely small dt.
      last_time = (float)data_packet.data.data_packet.timestamp_seconds - 0.005F;
    }

    if (cmd_status != TASKCMD_SETUP && cmd_status != TASKCMD_MOCK_SETUP) {
      // setting the current data packet to the instance of data to be used in this next filter
      // update
      raw_data_instance.timestamp_seconds = data_packet.data.data_packet.timestamp_seconds;
      memcpy(&raw_data_instance.pressure_pascals, &data_packet.data.data_packet.pressure_pascals,
             sizeof(raw_data_instance) - sizeof(raw_data_instance.timestamp_seconds));
      float dt = (float)raw_data_instance.timestamp_seconds - last_time;

      if (dt > 1e-6) {
        last_time = (float)raw_data_instance.timestamp_seconds;
        // build control input vector
        float u[ESKF_CONTROL_DIM] = {raw_data_instance.raw_acceleration_x_gs,
                                     raw_data_instance.raw_acceleration_y_gs,
                                     raw_data_instance.raw_acceleration_z_gs,
                                     raw_data_instance.raw_angular_rate_x_deg_per_s,
                                     raw_data_instance.raw_angular_rate_y_deg_per_s,
                                     raw_data_instance.raw_angular_rate_z_deg_per_s};

        // predict
        eskf_predict(&eskf, u, dt);

        // build measurement vector and update
        float z_raw[ESKF_MEASUREMENT_DIM] = {
            raw_data_instance.pressure_pascals,
            raw_data_instance.magnetic_field_x_microteslas,
            raw_data_instance.magnetic_field_y_microteslas,
            raw_data_instance.magnetic_field_z_microteslas,
        };
        //eskf_set_measurement(&eskf, z_raw);
        //eskf_update(&eskf);

        // Copy estimates back to data packet
        memcpy(&data_packet.data.data_packet.est_position_z_meters, eskf.x_nom,
               ESKF_NOMINAL_DIM * sizeof(float));

        // wait for all sensors to collect data
        xEventGroupWaitBits(sensors_collected,
                            BMP581_TASK_BIT | ICM45686_TASK_BIT |
                                MMC5983MA_TASK_BIT,
                            pdTRUE, // clear bits after unblocking
                            pdTRUE, // wait for ALL bits
                            portMAX_DELAY);
      }
    }
  }
}
