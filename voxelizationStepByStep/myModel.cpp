#include "myModel.h"

myModel::myModel(std::string path) {
	loadModel(path);
	//simplify();
	optimize();
	//modelChange();
	makeAABB();
}

void myModel::loadModel(std::string path) {

	Assimp::Importer import;
	const aiScene* scene = import.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace);

	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
		std::cout << "ERROR::ASSIMP::" << import.GetErrorString() << std::endl;
		return;
	}

	directory = path.substr(0, path.find_last_of('/'));
	processNode(scene->mRootNode, scene);

}

//һ��node����mesh����node��������Ҫ�ݹ飬�����е�mesh���ó���
//���е�ʵ�����ݶ���scene�У���node�д洢����scene������
void myModel::processNode(aiNode* node, const aiScene* scene) {

	for (uint32_t i = 0; i < node->mNumMeshes; i++) {
		aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
		this->meshs.push_back(processMesh(mesh, scene));
	}

	for (uint32_t i = 0; i < node->mNumChildren; i++) {
		processNode(node->mChildren[i], scene);
	}

}

Mesh myModel::processMesh(aiMesh* mesh, const aiScene* scene) {

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	std::vector<Texture> textures;

	for (uint32_t i = 0; i < mesh->mNumVertices; i++) {

		Vertex vertex;
		glm::vec3 vector;

		vector.x = mesh->mVertices[i].x;
		vector.y = mesh->mVertices[i].y;
		vector.z = mesh->mVertices[i].z;
		vertex.pos = vector;

		if (mesh->HasNormals()) {

			vector.x = mesh->mNormals[i].x;
			vector.y = mesh->mNormals[i].y;
			vector.z = mesh->mNormals[i].z;
			vertex.normal = vector;

		}

		if (mesh->HasTangentsAndBitangents()) {

			vector.x = mesh->mTangents[i].x;
			vector.y = mesh->mTangents[i].y;
			vector.z = mesh->mTangents[i].z;
			vertex.tangent = vector;

		}

		if (mesh->mTextureCoords[0]) // �����Ƿ����������ꣿ
		{
			glm::vec2 vec;
			vec.x = mesh->mTextureCoords[0][i].x;
			vec.y = mesh->mTextureCoords[0][i].y;
			vertex.texCoord = vec;
		}
		else {
			vertex.texCoord = glm::vec2(0.0f, 0.0f);
		}

		vertices.push_back(vertex);

	}

	for (uint32_t i = 0; i < mesh->mNumFaces; i++) {
		aiFace face = mesh->mFaces[i];
		for (uint32_t j = 0; j < face.mNumIndices; j++) {
			indices.push_back(face.mIndices[j]);
		}
	}

	Material mat;
	if (mesh->mMaterialIndex >= 0) {
		
		aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
		aiColor3D color;
		material->Get(AI_MATKEY_COLOR_AMBIENT, color);
		mat.bxdfPara = glm::vec4(color.r, color.g, color.b, 1.0);
		material->Get(AI_MATKEY_COLOR_DIFFUSE, color);
		mat.kd = glm::vec4(color.r, color.g, color.b, 1.0);
		material->Get(AI_MATKEY_COLOR_SPECULAR, color);
		mat.ks = glm::vec4(color.r, color.g, color.b, 1.0);
		material->Get(AI_MATKEY_COLOR_EMISSIVE, color);
		mat.ke = glm::vec4(color.r, color.g, color.b, 1.0);

		std::vector<Texture> diffuseMaps = loadMaterialTextures(material, aiTextureType_DIFFUSE, "texture_albedo");
		textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());

		//std::vector<Texture> specularMaps = loadMaterialTextures(material, aiTextureType_SPECULAR, "texture_specular");
		//textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());

		std::vector<Texture> normalMaps = loadMaterialTextures(material, aiTextureType_HEIGHT, "texture_normal");
		textures.insert(textures.end(), normalMaps.begin(), normalMaps.end());

	}

	return Mesh(vertices, indices, textures, mat);

}

std::vector<Texture> myModel::loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName) {

	std::vector<Texture> textures;
	for (unsigned int i = 0; i < mat->GetTextureCount(type); i++)
	{
		aiString str;
		mat->GetTexture(type, i, &str);
		bool skip = false;
		for (unsigned int j = 0; j < textures_loaded.size(); j++)
		{
			if (std::strcmp(textures_loaded[j].path.data(), str.C_Str()) == 0)
			{
				textures.push_back(textures_loaded[j]);
				skip = true;
				break;
			}
		}
		if (!skip)
		{   // �������û�б����أ��������
			Texture texture;
			//texture.id = TextureFromFile(str.C_Str(), directory);
			texture.type = typeName;
			texture.path = directory + '/' + str.C_Str();
			textures.push_back(texture);
			textures_loaded.push_back(texture); // ��ӵ��Ѽ��ص�������
		}
	}

	return textures;

}

void myModel::simplify() {

	std::vector<Mesh> simpleMeshs;
	for (int i = 0; i < this->meshs.size(); i++) {
		if (this->meshs[i].indices.size() < 100) {	//2950
			simpleMeshs.push_back(this->meshs[i]);
		}
	}
	this->meshs = simpleMeshs;
}

//obj�кܶ�ʱ����indices��vertices��һ���ģ���ô���ǿ����Ż���ȥ���ظ��Ķ��㣬ʹ��vertexBuffer����
//bear_box���model���߿����Ż����ˣ�vertice��indices������ͬ����ʵ����û��
void myModel::optimize() {

	for (int i = 0; i < this->meshs.size(); i++) {
		std::unordered_map<Vertex, uint32_t> uniqueVerticesMap{};
		std::vector<Vertex> uniqueVertices;
		std::vector<uint32_t> uniqueIndices;
		for (uint32_t j = 0; j < this->meshs[i].indices.size(); j++) {
			Vertex vertex = this->meshs[i].vertices[this->meshs[i].indices[j]];
			if (uniqueVerticesMap.count(vertex) == 0) {
				uniqueVerticesMap[vertex] = static_cast<uint32_t>(uniqueVertices.size());
				uniqueVertices.push_back(vertex);
			}
			uniqueIndices.push_back(uniqueVerticesMap[vertex]);
		}
		this->meshs[i].vertices = uniqueVertices;
		this->meshs[i].indices = uniqueIndices;
	}

}

//���ڵ�ģ��̫С�ˣ��Ŵ�10������
void myModel::modelChange() {

	/*
	glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(5.0f, 5.0f, 5.0f));
	for (int i = 0; i < this->meshs.size(); i++) {
		for (int j = 0; j < this->meshs[i].vertices.size(); j++) {
			glm::vec3 pos = this->meshs[i].vertices[j].pos;
			glm::vec4 changePos = model * glm::vec4(pos, 1.0f);
			this->meshs[i].vertices[j].pos = glm::vec3(changePos.x, changePos.y, changePos.z);
		}
	}
	*/

	for (int i = 0; i < this->meshs.size(); i++) {
		if (this->meshs[i].vertices.size() > 100) {

			glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(0.6f, -0.4f, 0.6f));
			for (int j = 0; j < this->meshs[i].vertices.size(); j++) {

				this->meshs[i].vertices[j].pos = glm::vec3(model * glm::vec4(this->meshs[i].vertices[j].pos, 1.0f));

			}

		}
	}

}

void myModel::makeAABB() {

	for (int i = 0; i < this->meshs.size(); i++) {

		//left right xyz
		AABBBox AABB = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
		for (int j = 0; j < this->meshs[i].vertices.size(); j++) {
			glm::vec3 worldPos = this->meshs[i].vertices[j].pos;
			AABB.leftX  = worldPos.x  < AABB.leftX  ? worldPos.x : AABB.leftX ;
			AABB.rightX = worldPos.x  > AABB.rightX ? worldPos.x : AABB.rightX;
			AABB.leftY  = worldPos.y  < AABB.leftY  ? worldPos.y : AABB.leftY ;
			AABB.rightY = worldPos.y  > AABB.rightY ? worldPos.y : AABB.rightY;
			AABB.leftZ  = worldPos.z  < AABB.leftZ  ? worldPos.z : AABB.leftZ ;
			AABB.rightZ = worldPos.z  > AABB.rightZ ? worldPos.z : AABB.rightZ;
		}
		//�����棬���Ǹ���0.2�Ŀ��
		if (AABB.leftX == AABB.rightX) {
			AABB.leftX = AABB.leftX - 0.01;
			AABB.rightX = AABB.rightX + 0.01;
		}
		if (AABB.leftY == AABB.rightY) {
			AABB.leftY = AABB.leftY - 0.01;
			AABB.rightY = AABB.rightY + 0.01;
		}
		if (AABB.leftZ == AABB.rightZ) {
			AABB.leftZ = AABB.leftZ - 0.01;
			AABB.rightZ = AABB.rightZ + 0.01;
		}
		this->meshs[i].AABB = AABB;

	}

}