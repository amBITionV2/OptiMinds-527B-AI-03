package com.saharsh.Code.editor.Platform.Analyze;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.NullNode;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

@Service
public class ProcessingService {

    private final ResultRepository repo;
    private final ObjectMapper mapper;

    public ProcessingService(ResultRepository repo, ObjectMapper mapper) {
        this.repo = repo;
        this.mapper = mapper;
    }

    /**
     * Returns:
     * data = [
     *   {
     *     "phone":  { person, electronic_devices, activity, image? },
     *     "laptop": { face_detection, gaze_detection, multiple_faces, identity_verification,
     *                 liveness_detection, device_detection, image? },
     *     "overall_risk_score": <int>,
     *     "violation": true
     *   },
     *   ...
     * ]
     * Only violation records are included. Images are embedded on BOTH sides for each violation.
     */
    public ProcessResponse process(long userId, long testId) {
        var phoneRows  = repo.fetchPhoneRows(userId, testId);
        var laptopRows = repo.fetchLaptopRows(userId, testId);

        // index laptop rows by timestamp (if duplicate timestamps, keep the first)
        Map<Long, ResultRepository.Row> laptopByTime = laptopRows.stream()
                .collect(Collectors.toMap(r -> r.time, r -> r, (a, b) -> a));

        List<PhoneLaptopRecord> items = new ArrayList<>();

        for (var p : phoneRows) {
            var l = laptopByTime.get(p.time);
            if (l == null) continue; // only matched times

            // Build a temporary MatchedItem to reuse existing scoring/violation logic
            MatchedItem temp = new MatchedItem();
            temp.task_id = p.testId;
            temp.userId  = p.userId;
            temp.time    = p.time;

            // Laptop-side fields (objects expected from JSON)
            temp.face_detection        = extractNode(l.result, "face_detection");
            temp.gaze_detection        = extractNode(l.result, "gaze_detection");
            temp.multiple_faces        = extractNode(l.result, "multiple_faces");
            temp.identity_verification = extractNode(l.result, "identity_verification");
            temp.liveness_detection    = normalizeLiveness(extractNode(l.result, "liveness_detection"));
            temp.device_detection      = extractNode(l.result, "device_detection");

            // Phone-side fields
            temp.person             = extractNode(p.result, "person");
            temp.electronic_devices = extractNode(p.result, "electronic_devices");
            temp.activity           = extractNode(p.result, "activity");

            // compute risk + violation
            int risk       = computeRisk(temp);
            boolean isViol = isViolation(temp);

            // only keep violations
            if (!isViol) continue;

            // build nested "phone" object
            var phoneObj = mapper.createObjectNode();
            phoneObj.set("person",             temp.person);
            phoneObj.set("electronic_devices", temp.electronic_devices);
            phoneObj.set("activity",           temp.activity);

            // build nested "laptop" object
            var laptopObj = mapper.createObjectNode();
            laptopObj.set("face_detection",        temp.face_detection);
            laptopObj.set("gaze_detection",        temp.gaze_detection);
            laptopObj.set("multiple_faces",        temp.multiple_faces);
            laptopObj.set("identity_verification", temp.identity_verification);
            laptopObj.set("liveness_detection",    temp.liveness_detection);
            laptopObj.set("device_detection",      temp.device_detection);

            // include synchronized images for BOTH sides (same timestamp)
            String phoneImgB64  = safeDataUrl(p.base64Image);
            String laptopImgB64 = safeDataUrl(l.base64Image);
            if (phoneImgB64 != null)  phoneObj.put("image",  phoneImgB64);
            if (laptopImgB64 != null) laptopObj.put("image", laptopImgB64);

            PhoneLaptopRecord rec = new PhoneLaptopRecord();
            rec.phone               = phoneObj;
            rec.laptop              = laptopObj;
            rec.overall_risk_score  = risk;
            rec.violation           = true; // by definition here

            items.add(rec);
        }

        int total = items.stream().mapToInt(i -> i.overall_risk_score).sum();

        ProcessResponse resp = new ProcessResponse();
        resp.result = true;
        resp.message = "Processed " + items.size() + " violation records.";
        resp.data = items; // data is List<PhoneLaptopRecord>
        resp.total_risk_score = total;
        return resp;
    }

    public ProcessResponse clearData() {
        ProcessResponse resp = new ProcessResponse();
        resp.result = true;
        resp.message = "Cleared server-side matched data cache.";
        resp.data = Collections.emptyList();
        resp.total_risk_score = 0;
        return resp;
    }

    // ---- helpers ----

    private JsonNode extractNode(JsonNode root, String field) {
        if (root == null || root.isNull()) return NullNode.getInstance();
        JsonNode n = root.get(field);
        return n == null ? NullNode.getInstance() : n;
    }

    // liveness_detection can be a JSON string -> parse it to object
    private JsonNode normalizeLiveness(JsonNode node) {
        if (node == null || node.isNull()) return NullNode.getInstance();
        if (node.isTextual()) {
            String txt = node.asText();
            try {
                return mapper.readTree(txt);
            } catch (Exception ignored) {}
        }
        return node;
    }

    private int computeRisk(MatchedItem item) {
        int total = 0;

        JsonNode faceDetection = item.face_detection;
        JsonNode gazeDetection = item.gaze_detection;
        JsonNode multipleFaces = item.multiple_faces;
        JsonNode identityVer   = item.identity_verification;
        JsonNode liveness      = item.liveness_detection;
        JsonNode person        = item.person;
        JsonNode electronic    = item.electronic_devices;
        JsonNode activity      = item.activity;

        boolean lookingAtScreen =
                (gazeDetection != null && "Looking at Screen".equals(text(gazeDetection.get("gaze_status"))))
                        || (activity != null && bool(activity.get("looking_at_screen")));
        if (!lookingAtScreen) total += 5;

        if (!"valid".equals(text(faceDetection.get("status"))) ) total += 3;
        if (!"One Face Detected".equals(text(multipleFaces.get("multiple_faces")))) total += 3;
        if (!"Same Face".equals(text(identityVer.get("identity_status")))) total += 3;
        if (!"Real Face Detected".equals(text(liveness.get("liveness_status")))) total += 3;

        if (!("valid".equals(text(person.get("status"))) && "One person detected.".equals(text(person.get("message")))))
            total += 3;

        if (!( "valid".equals(text(electronic.get("status"))) &&
                "Only one laptop detected.".equals(text(electronic.get("message"))) ))
            total += 2;

        if (!(activity != null && bool(activity.get("sitting")))) total += 5;

        return total;
    }

    private boolean isViolation(MatchedItem item) {
        JsonNode faceDetection = item.face_detection;
        JsonNode gazeDetection = item.gaze_detection;
        JsonNode multipleFaces = item.multiple_faces;
        JsonNode identityVer   = item.identity_verification;
        JsonNode liveness      = item.liveness_detection;
        JsonNode person        = item.person;
        JsonNode electronic    = item.electronic_devices;
        JsonNode activity      = item.activity;

        boolean cond =
                !( "Looking at Screen".equals(text(gazeDetection.get("gaze_status"))) ||
                        bool(activity.get("looking_at_screen")) ) ||
                        !"valid".equals(text(faceDetection.get("status"))) ||
                        !"One Face Detected".equals(text(multipleFaces.get("multiple_faces"))) ||
                        !"Same Face".equals(text(identityVer.get("identity_status"))) ||
                        !"Real Face Detected".equals(text(liveness.get("liveness_status"))) ||
                        !("valid".equals(text(person.get("status"))) && "One person detected.".equals(text(person.get("message")))) ||
                        !( "valid".equals(text(electronic.get("status"))) &&
                                "Only one laptop detected.".equals(text(electronic.get("message"))) ) ||
                        !bool(activity.get("sitting"));

        return cond;
    }

    private String text(JsonNode n) { return n == null || n.isNull() ? null : n.asText(); }
    private boolean bool(JsonNode n) { return n != null && !n.isNull() && n.asBoolean(false); }

    // Optionally add a data URL prefix if FE expects it; otherwise return raw base64.
    private String safeDataUrl(String b64) {
        if (b64 == null || b64.isBlank()) return null;
        // If your React viewer needs data URLs, uncomment the next line and choose the right MIME:
        // return "data:image/jpeg;base64," + b64;
        return b64; // raw Base64 is fine if FE already handles it
    }

    /** Each array element has only these four keys. */
    public static class PhoneLaptopRecord {
        public JsonNode phone;
        public JsonNode laptop;
        public int overall_risk_score; // exposed per-item score
        public boolean violation;      // exposed per-item violation flag (always true here)
    }
}
