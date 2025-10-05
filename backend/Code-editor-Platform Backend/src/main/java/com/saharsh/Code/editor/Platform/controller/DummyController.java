package com.saharsh.Code.editor.Platform.controller;

import com.saharsh.Code.editor.Platform.service.ExecService;
import org.apache.kafka.common.protocol.types.Field;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/dummy")
public class DummyController {
    @Autowired
    ExecService execService;
    @PostMapping("/produce")
    public void kafkaproduce(@RequestParam String userId,@RequestParam String testId) {
        execService.sendIds(userId,testId);
    }
}
