package com.saharsh.Code.editor.Platform.Analyze;

import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
@CrossOrigin
public class ProcessingController {

    private final ProcessingService service;

    public ProcessingController(ProcessingService service) {
        this.service = service;
    }

    public static class ProcessRequest {
        public long user_id;
        public long test_id;
    }

    @PostMapping("/process-matched-data")
    public ProcessResponse process(@RequestBody(required = false) ProcessRequest req) {
        long userId = (req != null ? req.user_id : 2L);
        long testId = (req != null ? req.test_id : 2L);
        return service.process(userId, testId);
    }

    @PostMapping("/clear-data")
    public ProcessResponse clear() {
        return service.clearData();
    }
}
