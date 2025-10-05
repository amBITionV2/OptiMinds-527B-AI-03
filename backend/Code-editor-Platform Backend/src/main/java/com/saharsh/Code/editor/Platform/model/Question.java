package com.saharsh.Code.editor.Platform.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.CreationTimestamp;

import java.sql.Timestamp;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Entity
public class Question{


    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer problemId;

    private String title;

    @Column(length = 1000)
    private String description;

    private String difficulty;

    private String TimeComplexity;

    private String  SpaceComplexity;

    @CreationTimestamp
    private Timestamp created_at;

}



