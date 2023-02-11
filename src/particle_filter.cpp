/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random> 
#include <string>
#include <vector>
#include <float.h>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::uniform_real_distribution;
using std::uniform_int_distribution;
using std::cout;
using std::endl;

// Initialize random number engine that generates pseudo-random numbers (i.e., derived from a known starting point)
std::default_random_engine gen;
// Initialize random number engine that produces non-deterministic random numbers
std::random_device r;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // Set the number of particles
  
  // Initialize normal distirubtions for x, y and theta, where mean is GPS value and standard deviation is sensor noise
  normal_distribution<double> dist_x(x, std[0]);   // This line creates a normal (Gaussian) distribution for x
  normal_distribution<double> dist_y(y, std[1]);   // This line creates a normal (Gaussian) distribution for y 
  normal_distribution<double> dist_theta(theta, std[2]);   // This line creates a normal (Gaussian) distribution for theta
  
  // Generate random initial particles
  for (int i = 0; i < num_particles; i++){
  	Particle new_particle;
    new_particle.id = i;
    new_particle.x = dist_x(gen);
    new_particle.y = dist_y(gen);
    new_particle.theta = dist_theta(gen);
    new_particle.weight = 1.0;
    particles.push_back(new_particle);
    weights.push_back(new_particle.weight);
    //cout << "new_particle.id " << new_particle.id << " new_particle.x " << new_particle.x << " new_particle.y " << new_particle.y << " new_particle.theta " << new_particle.theta << endl;
  }
  
  is_initialized = true;  // Set initialization done

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
    
  // Initialize normal distirubtions for x, y and theta, where mean is zero and standard deviation is sensor noise
  normal_distribution<double> dist_x(0.0, std_pos[0]);   // This line creates a normal (Gaussian) distribution for x
  normal_distribution<double> dist_y(0.0, std_pos[1]);   // This line creates a normal (Gaussian) distribution for y 
  normal_distribution<double> dist_theta(0.0, std_pos[2]);   // This line creates a normal (Gaussian) distribution for theta
  
  // Predict the state of particles (or car) based on velocity and yaw rate measurements at the next time step following bicycle motion model
  for (int i = 0; i < num_particles; i++){
    if (yaw_rate == 0.0){ // if yaw rate is zero
    	particles[i].x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
        particles[i].y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
        particles[i].theta = particles[i].theta;
    }
    else{ // if yaw rate is not zero
    	particles[i].x = particles[i].x + (velocity / yaw_rate) * (sin(particles[i].theta + delta_t * yaw_rate) - sin(particles[i].theta));
        particles[i].y = particles[i].y + (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + delta_t * yaw_rate));
        particles[i].theta = particles[i].theta + delta_t * yaw_rate;
    }
      
  	// add velocity and yaw measurement noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
  //cout << "Prediction step completed!!" << endl;
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (unsigned int i = 0; i < observations.size(); i++){
    int min_idx = -1; // Initialize Id of matching (or closest) predicted landmark to some random number
    double min_d = DBL_MAX; // Initialize minimum distance to matching (or closest) predicted landmark to a large value
    
    // Identify Id of matching (or closest) predicted landmark from the current transformed observation
    for (unsigned int j = 0; j < predicted.size(); j++){
      // Obatain the distance between current transformed observation and each predicted landmark
      double d = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      // Update Id of matching (or closest) predicted landmark, if jth predicted landmark is closer to the previously identified predicted landmark
      if (d < min_d) {
        min_d = d;
        min_idx = predicted[j].id;
      }
    }
    observations[i].id = min_idx; // set Id of matching (or closest) predicted landmark
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
   for (int i = 0; i < num_particles; i++){
     // Vector to hold predicted landmarks, i.e., map landmarks within sensor_range of particle (or predicted state)
     vector<LandmarkObs> predicted_landmarks;
     // Identify or predicte map landmarks within sensor_range of particle (or predicted state)
     for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++){
       // Obtain the distance between particle (or predicted state) and each map landmark
       double d = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
       if (d < sensor_range){ // if distance is less than the sensor range then add map landmark to the set of predicted landmarks
         LandmarkObs pred_landmark;
         pred_landmark.id = map_landmarks.landmark_list[j].id_i;
         pred_landmark.x = map_landmarks.landmark_list[j].x_f;
         pred_landmark.y = map_landmarks.landmark_list[j].y_f;
         predicted_landmarks.push_back(pred_landmark);
       }
     }
     
     // Vector to hold transformed observation mesurements from from car coordinate system to map coordinate system with respect to each particle 
     vector<LandmarkObs> transformed_obs;
     // Transform observation measurements from car coordinate system to map coordinate system with respect to each particle 
     for(unsigned int j = 0; j < observations.size(); j++){
       LandmarkObs obs_temp;
       obs_temp.x = particles[i].x + cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * observations[j].y;
       obs_temp.y = particles[i].y + sin(particles[i].theta) * observations[j].x + cos(particles[i].theta) * observations[j].y;
       transformed_obs.push_back(obs_temp);
     }
     // Identify association of transformed observation measurements to predicted lankmarks
     dataAssociation(predicted_landmarks, transformed_obs);
     
     // Update weight of the particle based on the landmark association
     double new_weight = 1.0; // Initialize new weight  to 1 to obtain updated weight
     double normalizer = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]); // Obtain Gaussian normalizer
     double x_sigma2 = std_landmark[0] * std_landmark[0]; // Obtain variance in x measurement
     double y_sigma2 = std_landmark[1] * std_landmark[1]; // Obtain variance in y measurement
     // Calculate likelihood of each transformed observation with associated predicted landmark using multivariate Gaussian probability density function
     for(unsigned int j = 0; j < transformed_obs.size(); j++){
       // find the predicted landmark associated with jth transformed observation
       int match_id;
       for(unsigned int k = 0; k < predicted_landmarks.size(); k++){
         if (transformed_obs[j].id == predicted_landmarks[k].id){ // If true, then kth predicted landmark is associated with jth transformed observation
           match_id = k;
           break;
         }
       }
       // x measurement part of multivariate Gaussian probability density function
       double x_measurement = (transformed_obs[j].x - predicted_landmarks[match_id].x) * (transformed_obs[j].x - predicted_landmarks[match_id].x) / (2.0 * x_sigma2);
       // y measurement part of multivariate Gaussian probability density function
       double y_measurement = (transformed_obs[j].y - predicted_landmarks[match_id].y) * (transformed_obs[j].y - predicted_landmarks[match_id].y) /  (2.0 * y_sigma2);
       new_weight *=  normalizer * exp(-(x_measurement + y_measurement));
     }
     // Update the weight of particle with new weight
     particles[i].weight = new_weight;
   }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  // Initialize a vector to hold resampled particles
  vector<Particle> resampled_particles;
  // Initialize uniform distirubtion to draw a random integer between 0 to num_particles - 1
  uniform_int_distribution<int> distInt(0, num_particles - 1);
  // Draw a random particle index
  int index = distInt(gen);
  // Initialize beta function
  double beta = 0.0;
  // Obtain maximum weight among the particles
  double max_weight = DBL_MIN;
  for (int i = 0; i < num_particles; i++){
    if (max_weight < particles[i].weight){
      max_weight = particles[i].weight;
    }
  }
  // Initialize uniform distirubtion to draw a random real  number between 0 to 2 * max_weight
  uniform_real_distribution<double> distReal(0.0, 2 * max_weight);
  // Resample num_particles 
  for (int i = 0; i < num_particles; i++){
    // Add random value between 0.0 to 2 * max_weight to the beta function
    beta += distReal(gen);
    while (particles[index].weight < beta){
      beta -= particles[index].weight; // subtract particles[index].weight from beta function
      index = (index + 1) % num_particles; // Increment the index by 1 such that it remains always between 0 to num_particles - 1
    }
    // Add particle represented by index 'index'
    resampled_particles.push_back(particles[index]);
  }
  particles.clear();
  particles = resampled_particles;

}

/*void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
 
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}*/
