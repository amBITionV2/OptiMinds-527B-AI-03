package com.saharsh.Code.editor.Platform.model;



import jakarta.persistence.*;
import java.time.OffsetDateTime;

@Entity
@Table(name = "user_images")
public class UserImage {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    // if you don't have a Users entity relationship, keep userId as column
    @Column(name = "user_id", nullable = false)
    private Long userId;


    @Lob
    @Basic(fetch = FetchType.LAZY)
    @Column(name = "image_data", nullable = false)
    private byte[] imageData;      // <-- MUST be byte[]



    public UserImage() {}

    // getters / setters

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }



    public byte[] getImageData() { return imageData; }
    public void setImageData(byte[] imageData) { this.imageData = imageData; }

}

