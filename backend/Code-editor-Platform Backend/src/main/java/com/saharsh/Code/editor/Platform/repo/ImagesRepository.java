package com.saharsh.Code.editor.Platform.repo;

import com.saharsh.Code.editor.Platform.model.Images;
import org.springframework.data.jpa.repository.JpaRepository;



import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ImagesRepository extends JpaRepository<Images, Long> {

}

