package com.saharsh.Code.editor.Platform.service;

import com.saharsh.Code.editor.Platform.model.Images;
import com.saharsh.Code.editor.Platform.model.UserImage;
import com.saharsh.Code.editor.Platform.model.Users;
import com.saharsh.Code.editor.Platform.repo.UserImageRepository;
import com.saharsh.Code.editor.Platform.repo.UsersRepository;
import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;

@Service
public class ImagesService {

    private static final int MAX_IMAGES_PER_USER = 5;

    @Autowired
    private UserImageRepository userImageRepository;

    @Autowired
    private UsersRepository usersRepository;

    public int count(Long userId) {
        return (int) userImageRepository.countByUserId(userId);
    }

    public int saveImage(Long userId,MultipartFile frame) throws Exception {
        int x=count(userId);
        if(x<5) {
            if (frame.isEmpty()) throw new Exception("No image provided");
            byte[] imageBytes = frame.getBytes();

            UserImage image = new UserImage();
            image.setUserId(userId);
            image.setImageData(imageBytes); // <- must be byte[]

            userImageRepository.save(image);
            return x+1;
        }
        return 5;
    }
}
