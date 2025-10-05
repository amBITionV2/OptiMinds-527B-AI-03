package com.saharsh.Code.editor.Platform.controller;





import com.saharsh.Code.editor.Platform.model.Images;
import com.saharsh.Code.editor.Platform.model.UserImage;
import com.saharsh.Code.editor.Platform.service.ImagesService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ContentDisposition;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.*;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/images")
public class ImagesController {
    @Autowired
    private ImagesService imagesService;
    @GetMapping("/checkImage")
    private int checkImages(@RequestParam Long userId) {
        return imagesService.count(userId);
    }

    @PostMapping("/upload")
    public ResponseEntity<?> uploadImage(
            @RequestParam("userId") Long userId,
            @RequestParam("frame") MultipartFile frame
    ) {
        try {
            return ResponseEntity.ok(imagesService.saveImage(userId,frame));

        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(500).body("Failed to save image: " + e.getMessage());
        }
    }


}

