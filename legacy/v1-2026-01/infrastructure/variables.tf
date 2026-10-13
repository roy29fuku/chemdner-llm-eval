variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}

variable "machine_type" {
  description = "GCP Machine Type"
  type        = string
  default     = "a2-standard-4"
}

variable "gpu_type" {
  description = "GPU Type"
  type        = string
  default     = "nvidia-tesla-a100"
}

variable "gpu_count" {
  description = "Number of GPUs"
  type        = number
  default     = 1
}

variable "disk_size" {
  description = "Boot disk size in GB"
  type        = number
  default     = 100
}

variable "instance_name" {
  description = "Instance name"
  type        = string
  default     = "biobert-chemdner-training"
}
