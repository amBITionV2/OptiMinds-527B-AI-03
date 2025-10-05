package com.saharsh.Code.editor.Platform.repo;

import com.saharsh.Code.editor.Platform.model.UserImage;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface UserImageRepository extends JpaRepository<UserImage, Long> {
    List<UserImage> findByUserId(Long userId);
    long countByUserId(Long userId); // preferred for count
}
