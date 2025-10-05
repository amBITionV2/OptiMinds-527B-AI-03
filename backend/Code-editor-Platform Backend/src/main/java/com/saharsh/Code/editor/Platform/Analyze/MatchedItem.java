package com.saharsh.Code.editor.Platform.Analyze;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.List;

public class MatchedItem {
    public long task_id;
    public long userId;
    public long time;

    // laptop-side fields
    public JsonNode face_detection;
    public JsonNode gaze_detection;
    public JsonNode multiple_faces;
    public JsonNode identity_verification;
    public JsonNode liveness_detection;
    public JsonNode device_detection;

    // phone-side fields
    public JsonNode person;
    public JsonNode electronic_devices;
    public JsonNode activity;

    // results
    public int  risk_score;
    public boolean violation;

    // legacy fields (backward compatibility)
    public String results_image; // laptop image (legacy)
    public String phone_image;   // phone image  (legacy)

    // new structured list
    public List<CapturedImage> images; // populated only when violation == true
}
