use pyo3::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};

#[pyclass]
struct PresenceTracker {
    tracker: presence::PresenceTracker,
}

#[pymethods]
impl PresenceTracker {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(PresenceTracker {
            tracker: presence::PresenceTracker::new(),
        })
    }

    /// Save the state of the tracker to a file
    /// Args:
    ///   file_path (str): The path to the file
    /// Returns:
    ///  None
    #[pyo3(text_signature = "(self, file_path)")]
    fn save_to_file(&self, file_path: String) -> PyResult<()> {
        self.tracker.save_to_file(&file_path).unwrap();
        Ok(())
    }

    /// Load the state of the tracker from a file
    /// Args:
    ///    file_path (str): The path to the file    
    /// Returns:
    ///   None
    #[pyo3(text_signature = "(self, file_path)")]
    fn load_from_file(&mut self, file_path: String) -> PyResult<()> {
        self.tracker = presence::PresenceTracker::load_from_file(&file_path).unwrap();
        Ok(())
    }

    /// Update the tracker with a batch of detections
    /// Args:
    ///    current_time (int): The current time in microseconds
    ///    detections (List[Tuple[int, float]]): A list of tuples containing the class id and confidence
    ///    confidence_threshold (float): The confidence threshold
    /// Returns:
    ///    None
    #[pyo3(text_signature = "(self, current_time, detections, confidence_threshold)")]
    fn update_batch(
        &mut self,
        current_time: u64,
        detections: Vec<(u16, f32)>,
        confidence_threshold: f32,
    ) -> PyResult<()> {
        self.tracker
            .update_batch(current_time, &detections, confidence_threshold);
        Ok(())
    }

    /// Query the presence of a class in a time range
    /// Args:
    ///   class_id (int): The class id
    ///   start_time (int): The start time in microseconds
    ///   end_time (int): The end time in microseconds
    /// Returns:
    ///  float: The presence of the class in the time range
    #[pyo3(text_signature = "(self, class_id, start_time, end_time)")]
    fn query_presence(&self, class_id: u16, start_time: u64, end_time: u64) -> PyResult<f32> {
        self.tracker
            .query_presence(class_id, start_time, end_time)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))
    }

    /// Lists presence timestamps for a class within a time range
    ///
    /// Args:
    ///     class_id (int): Object class identifier
    ///     start_time (int): Start time in microseconds
    ///     end_time (int): End time in microseconds
    ///
    /// Returns:
    ///     List[int]: List of presence timestamps
    #[pyo3(text_signature = "(self, class_id, start_time, end_time)")]
    fn list_presence_timestamps(
        &self,
        class_id: u16,
        start_time: u64,
        end_time: u64,
    ) -> PyResult<Vec<u64>> {
        self.tracker
            .list_presence_times(class_id, start_time, end_time)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))
    }
}

#[pymodule]
fn _presence_pyo3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PresenceTracker>()?;
    Ok(())
}
