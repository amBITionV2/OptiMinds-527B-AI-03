package com.saharsh.Code.editor.Platform.service;



import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;

@Service
public class ExecService {

    private final KafkaTemplate<String, String> kafkaTemplate;

    @Value("${app.topic}")
    private String topic;

    public ExecService(KafkaTemplate<String, String> kafkaTemplate) {
        this.kafkaTemplate = kafkaTemplate;
    }
    public void sendIds(String id1, String id2) {
        String payload = id1 + ":" + id2;
        kafkaTemplate.send(topic, payload);
        System.out.println("Produced message: " + payload + " -> topic " + topic);
    }
}

