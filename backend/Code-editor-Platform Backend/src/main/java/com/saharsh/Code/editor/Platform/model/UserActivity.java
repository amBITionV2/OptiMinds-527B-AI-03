package com.saharsh.Code.editor.Platform.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;

@Data
@Entity
@NoArgsConstructor
@AllArgsConstructor
public class UserActivity {
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    public int id;

    private String type;
    private String phase;
    private Instant ts;
    private Long duration;

    @Lob
    private String details;

    @ManyToOne(fetch = FetchType.EAGER)
    @JoinColumn(name = "user_id", nullable = false)
    private Users user;

    @ManyToOne(fetch = FetchType.EAGER)
    @JoinColumn(name = "test_id", nullable = false)
    private Test test;
}

